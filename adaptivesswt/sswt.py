#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import imp
import logging
from multiprocessing import cpu_count
from typing import Optional, Tuple

import numpy as np
import pywt
from numba import njit, prange, set_num_threads

import adaptivesswt

from .utils.freq_utils import (
    calcFilterLength,
    calcScalesAndFreqs,
    getDeltaAndBorderFreqs,
    getDeltaScales,
)
from .utils.plot_utils import plot_cwt_filters

logger = logging.getLogger(__name__)

from . import __backend as backend

if backend == 'opencl':
    import pyopencl as cl
    import pyopencl.array as cl_array

    platforms = cl.get_platforms()

    ctx = cl.Context(
        dev_type=cl.device_type.ALL,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])

    queue = cl.CommandQueue(ctx)

    _freq_agregate_prg = cl.Program(ctx,
        """
        #define PYOPENCL_DEFINE_CDOUBLE
        #include <pyopencl-complex.h>
        __kernel void agregate(
            __global const double *deltaFreqs, __global const double *borderFreqs,
            __global const double *aScale,  __global const double *wab, __global const cdouble_t *tr_matr,
            __global cdouble_t *sst, __global int *width, __global int *height)
        {
            int wd = *width;
            int hg = *height;
            int r_gid = get_global_id(0);
            int c_gid = get_global_id(1);
            int idx = c_gid + wd*r_gid;
            /*
            if (wab[idx] >= borderFreqs[r_gid] && wab[idx] < borderFreqs[r_gid+1]){
                sst[idx] = tr_matr[idx];
            }*/
            for(int w=0; w<hg; w++){
                if (wab[c_gid+w*wd] >= borderFreqs[r_gid] && wab[c_gid+w*wd] < borderFreqs[r_gid+1]){
                    sst[idx] = cdouble_add(sst[idx], cdouble_mulr(tr_matr[c_gid+w*wd] , aScale[w] / deltaFreqs[r_gid]));
                }
            }
        }
        """)

    mf = cl.mem_flags

    try:
        _freq_agregate_prg.build()
    except Exception:
        print("Error:")
        print(_freq_agregate_prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
        raise
    freq_agregate_knl = _freq_agregate_prg.agregate  # Use this Kernel object for repeated calls

    def _freq_agregate_cl(deltaFreqs: np.ndarray, borderFreqs: np.ndarray,
                   aScale: np.ndarray, wab: np.ndarray, tr_matr: np.ndarray,
                   sst: np.ndarray):
        deltaFreqs_dev = cl_array.to_device(queue, deltaFreqs)
        borderFreqs_dev = cl_array.to_device(queue, borderFreqs)
        aScale_dev = cl_array.to_device(queue, aScale)
        wab_dev = cl_array.to_device(queue, wab)
        tr_matr_dev = cl_array.to_device(queue, tr_matr)
        sst_dev = cl_array.to_device(queue, sst)

        width_dev = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(sst_dev.shape[1])
            )
        height_dev = cl.Buffer(
            ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(sst_dev.shape[0])
        )
        freq_agregate_knl(queue, sst.shape, None, deltaFreqs_dev.data, borderFreqs_dev.data,
            aScale_dev.data, wab_dev.data, tr_matr_dev.data, sst_dev.data,
            width_dev, height_dev)

        queue.finish()
        sst = sst_dev.get()
        return sst

    _freq_agregate = _freq_agregate_cl

else:  #numba
    @njit(parallel=True, fastmath=True)
    def _freq_agregate_nb(deltaFreqs: np.ndarray, borderFreqs: np.ndarray,
                       aScale: np.ndarray, wab: np.ndarray, tr_matr: np.ndarray,
                       sst: np.ndarray):
        for b in prange(sst.shape[1]):        # Time
            for w in prange(sst.shape[0]):    # Frequency
                components = np.logical_and(wab[:,b] > borderFreqs[w],
                                            wab[:,b] <= borderFreqs[w+1])

                sst[w,b] = (tr_matr[components,b] * aScale[components]).sum() / deltaFreqs[w]
        return sst

    _freq_agregate = _freq_agregate_nb

def sswt(signal: np.ndarray,
         min_freq: float,
         max_freq: float,
         num_freqs: int,
         ts: float=1,
         threshold: float=1,
         wav: pywt.ContinuousWavelet=pywt.ContinuousWavelet('cmor1-0.5'),  # type: ignore # Pylance seems to fail finding ContinuousWavelet within pywt
         custom_scales: Optional[np.ndarray]=None,
         pad: int=0,
         num_processes: int=cpu_count(),
         plot_filters: bool=False,
         c_psi: Optional[complex]=None,
         **kwargs
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,  np.ndarray, np.ndarray]:
    """Calculates the Synchrosqueezed Wavelet Transform.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    min_freq : float
        Minimum analysis frequency (min = Fs/N).
    max_freq : float
        Maximum analysis frequency (max = Fs/2).
    num_freqs : float
        Number of analysis frequencies
    ts : float, optional
        Sample time, by default 1
    threshold : float, optional
        Remapping threshold of the synchrozqueeze process, by default 1
    wav : pywt.ContinuousWavelet, optional
        Family of wavelet to be used, by default 'cmor1-0.5'
    num_processes : int, optional
        Number of used sub-processes, by default 4
    custom_scales : np.ndarray, optional
        Custom scales for the WT, by default None
    pad : int, optional
        Number of padding samples for overlap-and-add applications,
        if negative calculated automatically based on wavelet max length, by default 0
    plot_filters : bool, optional
        Wavelet filters spectrum is ploted if True (affects performance), by default False
    c_psi: complex, optional
        Reconstruction coeficient. Needed if pad is different from 0.

    Returns
    -------
    tuple
        (St:np.ndarray, cwt:np.ndarray, freqs:np.ndarray, tail:np.ndarray)
        Matrces with SST and CWT; array with analysis frequencies; frequency remmaping matrix, and signal tail for overlap (tail is empty if pad is 0).
    """

    #####################
    # Wavelet Transform #
    #####################
    wcf = wav.center_frequency

    #### Scales ####
    if custom_scales is None:
        scales, freqs, delta_scales, _ = calcScalesAndFreqs(ts, wcf, min_freq, max_freq, num_freqs)
    else:
        scales = custom_scales
        logger.debug('Using custom scales')

        delta_scales = getDeltaScales(scales)

    logger.debug('Scales: \n%s\n', scales)
    logger.debug('Delta scales: \n%s\n', delta_scales)

    # padding for streaming:
    maxWavLen = calcFilterLength(wav, scales.max()) if (pad < 0) else pad
    signal = np.pad(signal, (0, maxWavLen), mode='constant')

    #### CWT ####
    if plot_filters: plot_cwt_filters(wav, scales, ts, signal)
    cwt, freqs = pywt.cwt(signal, scales, wav, sampling_period=ts, method='fft')

    #### Transforms ####
    match kwargs.get('transform', 'sst'):
        case 'tsst':
            St, wab = time_synchrosqueeze(cwt, freqs, ts, threshold, num_processes)
        case 'tfr':
            St, (wab, _) = tf_synchrosqueeze(cwt, freqs, ts, scales, delta_scales, threshold, num_processes)
        case tr:    # 'sst', 'set'
            St, wab = freq_synchrosqueeze(cwt, freqs, ts, scales, delta_scales, threshold, num_processes, transform=tr)
        # case _:    # 'sst'
        #    St, wab = freq_synchrosqueeze(cwt, freqs, ts, scales, delta_scales, threshold, num_processes)

    if pad != 0:
        assert c_psi is not None, 'Atention: c_psi is needed if pad is != 0'
        tail = reconstruct(St[:,-maxWavLen:], c_psi, freqs)
        lastIdx = -maxWavLen
    else:
        tail, lastIdx = np.array([]), None

    return St[:,:lastIdx], cwt[:,:lastIdx], freqs, wab[:,:lastIdx], tail


def get_freq_remapping(cwt: np.ndarray=np.array([[]]), threshold: float=0.1,
                       ts: float=1) -> np.ndarray:

    # Eq. (7) - "Adaptive synchrosqueezing based on a quilted short time
    # Fourier transform" - A. Berrian, N. Saito.
    # Exact estimator:

    cwt_p = np.roll(cwt, -1, axis=1)
    cwt_p[:,-1] = 0
    w_ab = np.angle(np.divide(cwt_p, cwt, out=np.zeros_like(cwt),
                              where=abs(cwt)>threshold)) / (2 * np.pi * ts)
    # Last term is added in order to convert from normalized omega to frecuency in Hz
    return w_ab

def get_time_remapping(cwt: np.ndarray=np.array([[]]), threshold: float=0.1,
                       time: np.ndarray=np.array([]), freqs: np.ndarray=np.array([])) -> np.ndarray:
      # Eq. (13) - "The Synchrosqueezing algorithm for time-varying spectral
    # analysis: robustness properties and new paleoclimate applications" -
    # G. Thakur, E. Brevdo, N. S. Fučkar, and Hau-Tieng Wu:
    #
    dCWT_t = np.gradient(cwt, freqs, axis=0)
    t_ab = np.zeros_like(cwt, dtype='float64') + time
    pos = abs(cwt) > threshold
    t_ab[pos] -= 1 * np.imag(dCWT_t[pos] / cwt[pos]) / (2* np.pi)
    t_ab[np.logical_not(pos)]=0

    return t_ab


def freq_synchrosqueeze(cwt_matr: np.ndarray, freqs: np.ndarray, ts: float, scales: np.ndarray,
                        delta_scales: np.ndarray, threshold: float,
                        num_processes: int, transform='sst') -> Tuple[np.ndarray, np.ndarray]:

    scaleExp = -3/2
    aScale = (scales ** scaleExp) * delta_scales
    logger.debug('a_k^{%s} * da_k: \n%s\n', scaleExp, aScale)

    #### Frecuencies ####
    deltaFreqs, borderFreqs = getDeltaAndBorderFreqs(freqs)

    logger.debug("CWT frequencies: \n%s\n", freqs)
    logger.debug("deltaFreqs: \n%s\n", deltaFreqs)
    logger.debug("Frequency band limits: \n%s\n", borderFreqs)

    # Map (a,b) -> (w(a,b), b)

    logger.info('Calculating instantaneous frequencies...')

    wab = get_freq_remapping(cwt_matr, threshold, ts)

    sst = np.zeros_like(cwt_matr)

    set_num_threads(num_processes)
    match transform:
        case 'set':
            _freq_extract(deltaFreqs, borderFreqs, aScale, wab, cwt_matr, sst)
    ####################################
    # Sychrosqueezing parallel process
    ####################################
        case _:
            sst = _freq_agregate(deltaFreqs, borderFreqs, aScale, wab, cwt_matr, sst)

    logger.info('Synchrosqueezing Done!')

    return sst, wab

def time_synchrosqueeze(cwt_matr: np.ndarray, freqs: np.ndarray, ts: float,
                        threshold: float, num_processes: int) -> Tuple[np.ndarray, np.ndarray]:

    time = np.linspace(0, ts*cwt_matr.shape[1], cwt_matr.shape[1], endpoint=False)

    # Map (a,b) -> (a, t(a, b))

    logger.info('Calculating instantaneous times...')

    tab = get_time_remapping(cwt_matr, threshold, time, freqs)

    tsst = np.zeros_like(cwt_matr)

    ####################################
    # Sychrosqueezing parallel process
    ####################################

    set_num_threads(num_processes)
    _time_agregate(ts, time, tab, cwt_matr, tsst)
    logger.info('Time-synchrosqueezing Done!')

    return tsst, tab

def tf_synchrosqueeze(cwt_matr: np.ndarray, freqs: np.ndarray, ts: float,
                      scales: np.ndarray, delta_scales: np.ndarray, threshold: float,
                      num_processes: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    #### Frecuencies ####
    scaleExp = -3/2
    aScale = (scales ** scaleExp) * delta_scales

    deltaFreqs, borderFreqs = getDeltaAndBorderFreqs(freqs)
    wab = get_freq_remapping(cwt_matr, threshold, ts)

    #### Times ####
    time = np.linspace(0, ts*cwt_matr.shape[1], cwt_matr.shape[1], endpoint=False)
    tab = get_time_remapping(cwt_matr, threshold, time, freqs)

    ####################################
    # Sychrosqueezing parallel process
    ####################################
    set_num_threads(num_processes)
    sst = np.zeros_like(cwt_matr)
    _freq_agregate(deltaFreqs, borderFreqs, aScale, wab, cwt_matr, sst)
    tfr = np.zeros_like(cwt_matr)
    _time_agregate(ts, time, tab, sst, tfr)

    return tfr, (wab, tab)

@njit(parallel=True, fastmath=True)
def _freq_extract(deltaFreqs: np.ndarray, borderFreqs: np.ndarray,
                  aScale: np.ndarray, wab: np.ndarray, tr_matr: np.ndarray,
                  sst: np.ndarray):
    for b in prange(sst.shape[1]):        # Time
        for w in prange(sst.shape[0]):    # Frequency
            if (wab[w,b] > borderFreqs[w]) and (wab[w,b] <= borderFreqs[w+1]):
                sst[w,b] = tr_matr[w,b]  #/ deltaFreqs[w]

@njit(parallel=True, fastmath=True)
def _time_agregate(ts: float, time: np.ndarray, tab: np.ndarray,
                   tr_matr: np.ndarray, tsst: np.ndarray):
    for w in prange(tsst.shape[0]):        # Frequency
        for b in prange(tsst.shape[1]):    # Time
            components = np.logical_and(tab[w,:] > time[b],
                                        tab[w,:] <= time[b+1])
            tsst[w,b] = (tr_matr[w,components]).sum() # / (2*np.pi)

def reconstruct(sst: np.ndarray, c_psi: complex,
                freqs: np.ndarray)-> np.ndarray:
    """Reconstruct signal from its SST

    Parameters
    ----------
    sst : np.ndarray
        SST Matrix
    wavelet : pywt.ContinuousWavelet
        Wavelet Family used in the SST

    Returns
    -------
    np.ndarray
        Reconstructed signal samples
    """
    deltaFreqs, _ = getDeltaAndBorderFreqs(freqs)
    signalR = (1/c_psi) * (sst * deltaFreqs[:,np.newaxis]).sum(axis=0)
    return signalR.real

def reconstruct_tsst(tsst: np.ndarray, c_psi: complex,
                     freqs: np.ndarray)-> np.ndarray:
    """Reconstruct signal from its TSST

    Parameters
    ----------
    sst : np.ndarray
        SST Matrix
    wavelet : pywt.ContinuousWavelet
        Wavelet Family used in the SST

    Returns
    -------
    np.ndarray
        Reconstructed signal samples
    """
    deltaFreqs, _ = getDeltaAndBorderFreqs(freqs)
    signalR = (1/c_psi) * (tsst * deltaFreqs[:,np.newaxis]).sum(axis=0)
    return signalR.real


def reconstructCWT(cwt: np.ndarray, wav: pywt.ContinuousWavelet, # type: ignore # Pylance seems to fail finding ContinuousWavelet within pywt
                   scales: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    psi_w, x_w = wav.wavefun(wav.upper_bound-wav.lower_bound)
    C_w = abs(psi_w[np.argmin(np.abs(x_w))]) * (wav.upper_bound / 4)
    signalR = (1/C_w) *  np.sum(cwt / (scales[:, np.newaxis]**0.5)  * np.exp(-1j*freqs/scales)[:,np.newaxis], axis=0)
    # TODO: check scaling factor
    return signalR.real


def main():
    logging.basicConfig(filename='sswt.log', filemode='w',
                        format='%(levelname)s - %(asctime)s - %(name)s:\n %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    import matplotlib
    import matplotlib.pyplot as plt

    font = {'family': 'normal', 'weight': 'normal', 'size': 14}
    matplotlib.rc('font', **font)
    plt.rcParams['text.usetex'] = True

    from .configuration import Configuration
    from .utils import signal_utils as generator
    from .utils.measures_utils import renyi_entropy

    stopTime = 12
    fs = 400
    signalLen = stopTime * fs

    t, step = np.linspace(0, stopTime, signalLen, endpoint=False, retstep=True)
    ts = float(step)  # type: ignore # np.Floating[Any] to float

    # signal = generator.testSine(t, 0.2) + generator.testSine(t,1) + generator.testSine(t, 5) + generator.testSine(t,10)
    # f, signal = generator.testSig(t)
    # f, signal = generator.crossChrips(t, 2, 8, 2)
    # f, signal = generator.testChirp(t, 3, 6)
    # _, signal = generator.quadraticChirp(t, 1, 30)
    f, signal = generator.dualQuadraticChirps(t, (8,5),(2,4))
    # signal = np.zeros_like(t)
    # signal[fs:2*fs]=1.0

    wcf = 1
    wbw = 4

    max_freq = 11
    min_freq = 0.1
    num_freqs = 64

    config = Configuration(
        min_freq=min_freq,
        max_freq=max_freq,
        num_freqs=num_freqs,
        ts=ts,
        wcf=wcf,
        wbw=wbw,
        wavelet_bounds=(-8,8),
        threshold=signal.max()/(100),
        plot_filters=False)

    wav = config.wav

    config.pad = 256

    scales, _, _, _ = calcScalesAndFreqs(ts, config.wcf, config.min_freq, config.max_freq, config.num_freqs)

    sst, cwt, freqs, wab, tail = sswt(signal, **config.asdict())
    rentrCWT = renyi_entropy(cwt,3)
    rentrSST = renyi_entropy(sst,3)
    print(f'Rènyi entropy of CWT = {rentrCWT}')
    print(f'Rènyi entropy of SST = {rentrSST}')


    mainFig = plt.figure('Comparación de métodos')
    gs = mainFig.add_gridspec(1, 3)
    mainAxes  = gs.subplots(sharex='col', sharey='row')
    mainFig.set_tight_layout(True)

    #mainAxes[0,0].plot(t[:len(signal)], signal)
    #mainAxes[0,0].set_title('Señal a analizar')


    mainAxes[1].pcolormesh(t, freqs, np.abs(cwt), cmap='viridis', shading='gouraud')
    mainAxes[1].set_title('Wavelet Transform')
    mainAxes[1].set_xlabel('t [s]', loc='right')
    mainAxes[1].set_ylabel('f [Hz]', loc='top')
    mainAxes[2].pcolormesh(t, freqs, np.abs(sst), cmap='viridis', shading='gouraud')
    mainAxes[2].set_title('Synchrosqueezing Transform')
    mainAxes[2].set_xlabel('t [s]', loc='right')
    mainAxes[2].set_ylabel('f [Hz]', loc='top')

    signalR_cwt = reconstructCWT(cwt, wav, scales, freqs)
    signalR_cwt /= signalR_cwt.max()
    signalR_sst = reconstruct(sst, config.c_psi, freqs)

    mainAxes[0].plot(t, f[0], label='Comp. 0')
    mainAxes[0].plot(t, f[1], label='Comp. 1')
    mainAxes[0].set_xlabel('t [s]', loc='right')
    mainAxes[0].set_ylabel('f [Hz]', loc='top')
    mainAxes[0].legend()
    mainAxes[0].set_title('Frecuencias intstantáneas')

    #### Transform size comparison ####

    config.num_freqs = 16
    sst16, _, freqs16, _, _ = sswt(signal, **config.asdict())
    config.num_freqs = 32
    sst32, _, freqs32, _, _ = sswt(signal, **config.asdict())
    config.num_freqs = 64
    sst64, _, freqs64, _, _ = sswt(signal, **config.asdict())
    config.num_freqs = 128
    sst128, _, freqs128, _, _ = sswt(signal, **config.asdict())

    sizeFig = plt.figure('Comparación de K')
    gsSize = sizeFig.add_gridspec(2, 2)
    sizeAxes  = gsSize.subplots(sharex='col', sharey='row')
    sizeFig.set_tight_layout(True)

    sizeAxes[0, 0].pcolormesh(t, freqs16, np.abs(sst16), cmap='viridis', shading='gouraud')
    sizeAxes[0, 0].set_title('K = 16')
    sizeAxes[0, 0].set_xlabel('t [s]', loc='right')
    sizeAxes[0, 0].set_ylabel('f [Hz]', loc='top')

    sizeAxes[0, 1].pcolormesh(t, freqs32, np.abs(sst32), cmap='viridis', shading='gouraud')
    sizeAxes[0, 1].set_title('K = 32')
    sizeAxes[0, 1].set_xlabel('t [s]', loc='right')
    sizeAxes[0, 1].set_ylabel('f [Hz]', loc='top')

    sizeAxes[1, 0].pcolormesh(t, freqs64, np.abs(sst64), cmap='viridis', shading='gouraud')
    sizeAxes[1, 0].set_title('K = 64')
    sizeAxes[1, 0].set_xlabel('t [s]', loc='right')
    sizeAxes[1, 0].set_ylabel('f [Hz]', loc='top')

    sizeAxes[1, 1].pcolormesh(t, freqs128, np.abs(sst128), cmap='viridis', shading='gouraud')
    sizeAxes[1, 1].set_title('K = 128')
    sizeAxes[1, 1].set_xlabel('t [s]', loc='right')
    sizeAxes[1, 1].set_ylabel('f [Hz]', loc='top')

    rentrSST16 = renyi_entropy(sst16,3)
    rentrSST32 = renyi_entropy(sst32,3)
    rentrSST64 = renyi_entropy(sst64,3)
    rentrSST128 = renyi_entropy(sst128,3)

    print(f'Rènyi entropy of SST with K=16 : {rentrSST16}')
    print(f'Rènyi entropy of SST with K=32 : {rentrSST32}')
    print(f'Rènyi entropy of SST with K=64 : {rentrSST64}')
    print(f'Rènyi entropy of SST with K=128 : {rentrSST128}')


    #### Transforms comparison ####
    tsstFig, tsstAx = plt.subplots(1,5)

    # signal = np.zeros_like(t)
    # pulse_width = 10
    # pulse_start, pulse_stop = int(len(t)//5), int(len(t)//5) + pulse_width
    # signal[pulse_start:pulse_stop]=1

    sst, cwt, freqs, tab, tail = sswt(signal, **config.asdict())
    tsstAx[0].pcolormesh(t, freqs, np.abs(cwt[:-1,:-1]), cmap='viridis', shading='flat') #'gouraud')
    tsstAx[0].set_title('Wavelet Transform')
    tsstAx[1].pcolormesh(t, freqs, np.abs(sst[:-1,:-1]), cmap='viridis', shading='flat') #'gouraud')
    tsstAx[1].set_title('Synchrosqueezing Transform')

    config.transform = 'tsst'
    tsst, cwt, freqs, tab, tail = sswt(signal, **config.asdict())
    tsstAx[2].pcolormesh(t, freqs, np.abs(tsst[:-1,:-1]), cmap='viridis', shading='flat') #'gouraud')
    tsstAx[2].set_title('Time synchrosqueezing Transform')

    config.transform = 'tfr'
    tfr, cwt, freqs, tab, tail = sswt(signal, **config.asdict())
    tsstAx[3].pcolormesh(t, freqs, np.abs(tfr)[:-1,:-1], cmap='viridis', shading='flat') #'gouraud')
    tsstAx[3].set_title('Time Frequency Reasignment')

    config.transform = 'set'
    tfr, cwt, freqs, tab, tail = sswt(signal, **config.asdict())
    tsstAx[4].pcolormesh(t, freqs, np.abs(tfr)[:-1,:-1], cmap='viridis', shading='flat') #'gouraud')
    tsstAx[4].set_title('Synchro-Extracting Transform')

    print(f'Max values: CWT={abs(cwt).max()}, SST={abs(sst).max()}, TSST={abs(tsst).max()}, TFR={abs(tfr).max()}\n')

    #### Timing ####
    import timeit
    passes = 10
    config.pad = 0
    time = timeit.timeit("lambda: sswt(signal, **config.asdict())", globals=globals(), number=passes)
    print(f'Excecution time for {passes} passes and {adaptivesswt.getBackend()} = {time}s')
    print(f'Execution time per signal second = {time / stopTime / passes} s/s')

    plt.show()


if __name__=='__main__':
    from . import setBackend
    setBackend('opencl')
    main()
