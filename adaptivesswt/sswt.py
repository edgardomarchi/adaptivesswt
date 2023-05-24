#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import multiprocessing as mp
from typing import Tuple

import numpy as np
import pywt
from numba import njit, prange, set_num_threads

from .utils.freq_utils import (
    calcFilterLength,
    calcScalesAndFreqs,
    getDeltaAndBorderFreqs,
    getDeltaScales,
)
from .utils.plot_utils import plotFilters

logger = logging.getLogger(__name__)

def sswt(signal: np.ndarray,
         minFreq: float,
         maxFreq: float,
         numFreqs: int,
         ts: float=1,
         threshold: float=1,
         wav: pywt.ContinuousWavelet=pywt.ContinuousWavelet('cmor1-0.5'),
         custom_scales: np.ndarray=None,
         pad: int=0,
         numProc: int=mp.cpu_count(),
         plotFilt: bool=False,
         C_psi: complex=None,
         **kwargs
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,  np.ndarray, np.ndarray]:
    """Calculates the Synchrosqueezed Wavelet Transform.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    minFreq : float
        Minimum analysis frequency (min = Fs/N).
    maxFreq : float
        Maximum analysis frequency (max = Fs/2).
    numFreqs : float
        Number of analysis frequencies
    ts : float, optional
        Sample time, by default 1
    threshold : float, optional
        Remapping threshold of the synchrozqueeze process, by default 1
    wav : pywt.ContinuousWavelet, optional
        Family of wavelet to be used, by default 'cmor1-0.5'
    numProc : int, optional
        Number of used sub-processes, by default 4
    custom_scales : np.ndarray, optional
        Custom scales for the WT, by default None
    pad : int, optional
        Number of padding samples for overlap-and-add applications,
        if negative calculated automatically based on wavelet max length, by default 0
    plotFilt : bool, optional
        Wavelet filters spectrum is ploted if True (affects performance), by default False
    C_psi: complex, optional
        Reconstruction coeficient. Needed if pad is different from 0.

    Returns
    -------
    tuple
        (St:np.ndarray, cwt:np.ndarray, freqs:np.ndarray, tail:np.ndarray)
        Matrces with SST and CWT; array with analysis frequencies; frequency remmaping matrix, and signal tail for overlap (tail is empty if pad is 0).
    """

    #####################
    #%% Wavelet Transform
    #####################
    wcf = wav.center_frequency

    #### Scales ####
    if custom_scales is None:
        scales, freqs, deltaScales, _ = calcScalesAndFreqs(ts, wcf, minFreq, maxFreq, numFreqs)
    else:
        scales = custom_scales
        logger.debug('Using custom scales')

        deltaScales = getDeltaScales(scales)

    logger.debug('Scales: \n%s\n', scales)
    logger.debug('Delta scales: \n%s\n', deltaScales)

    # padding for streaming:
    maxWavLen = calcFilterLength(wav, scales.max()) if (pad < 0) else pad
    signal = np.pad(signal, (0, maxWavLen), mode='constant')

    #### CWT ####
    if plotFilt: plotFilters(wav, scales, ts, signal)
    cwt, freqs = pywt.cwt(signal, scales, wav, sampling_period=ts,
                               method='fft')

    #### SSWT ####
    numbaParallel = kwargs.get('numbaParallel', True)

    match kwargs.get('transform', 'sst'):
        case 'tsst':
            St, wab = time_synchrosqueeze(cwt, freqs, ts, threshold, numProc)
        case 'tfr':
            St, wab = tf_synchrosqueeze(cwt, freqs, ts, scales, deltaScales, threshold, numProc)
        case _:    # 'sst'
            St, wab = freq_synchrosqueeze(cwt, freqs, ts, scales, deltaScales, threshold, numProc)

    if pad != 0:
        assert C_psi is not None, 'Atention: C_psi is needed if pad is != 0'
        tail = reconstruct(St[:,-maxWavLen:], C_psi, freqs)
        lastIdx = -maxWavLen
    else:
        tail, lastIdx = np.array([]), None

    return St[:,:lastIdx], cwt[:,:lastIdx], freqs, wab[:,:lastIdx], tail


def freq_synchrosqueeze(cwt_matr: np.ndarray, freqs: np.ndarray, ts: float, scales: np.ndarray,
                        deltaScales: np.ndarray, threshold: float,
                        numProc: int) -> Tuple[np.ndarray, np.ndarray]:

    scaleExp = -3/2
    aScale = (scales ** scaleExp) * deltaScales
    logger.debug('a_k^{%s} * da_k: \n%s\n', scaleExp, aScale)

    #### Frecuencies ####
    deltaFreqs, borderFreqs = getDeltaAndBorderFreqs(freqs)

    logger.debug("CWT frequencies: \n%s\n", freqs)
    logger.debug("deltaFreqs: \n%s\n", deltaFreqs)
    logger.debug("Frequency band limits: \n%s\n", borderFreqs)

    # Map (a,b) -> (w(a,b), b)

    logger.info('Calculating instantaneous frequencies...')
    # Eq. (7) - "Adaptive synchrosqueezing based on a quilted short time
    # Fourier transform" - A. Berrian, N. Saito.
    # Exact estimator:

    cwt_matr_p = np.roll(cwt_matr,-1, axis=1)
    cwt_matr_p[:,-1] = 0
    wab = np.angle(np.divide(cwt_matr_p, cwt_matr,
                             out=np.zeros_like(cwt_matr),
                             where=abs(cwt_matr)>threshold)) / (2 * np.pi * ts)
    # Last term is added in order to convert from normalized omega to frecuency in Hz

    sst = np.zeros_like(cwt_matr)

    ####################################
    # Sychrosqueezing parallel process
    ####################################
    set_num_threads(numProc)
    _freq_agregate(deltaFreqs, borderFreqs, aScale, wab, cwt_matr, sst)

    logger.info('Synchrosqueezing Done!')

    return sst, wab

def time_synchrosqueeze(cwt_matr: np.ndarray, freqs: np.ndarray, ts: float,
                        threshold: float, numProc: int) -> Tuple[np.ndarray, np.ndarray]:

    time = np.linspace(0, ts*cwt_matr.shape[1], cwt_matr.shape[1], endpoint=False)

    # Map (a,b) -> (a, t(a, b))

    logger.info('Calculating instantaneous times...')
    # Eq. (13) - "The Synchrosqueezing algorithm for time-varying spectral
    # analysis: robustness properties and new paleoclimate applications" -
    # G. Thakur, E. Brevdo, N. S. Fučkar, and Hau-Tieng Wu:
    #
    dCWT_t = np.gradient(cwt_matr, freqs, axis=0)
    tab = np.zeros_like(cwt_matr, dtype='float64') + time
    pos = abs(cwt_matr) > threshold
    tab[pos] -= 1 * np.imag(dCWT_t[pos] / cwt_matr[pos]) / (2* np.pi)
    tab[np.logical_not(pos)]=0

    tsst = np.zeros_like(cwt_matr)

    ####################################
    # Sychrosqueezing parallel process
    ####################################

    set_num_threads(numProc)
    _time_agregate(ts, time, tab, cwt_matr, tsst)
    logger.info('Time-synchrosqueezing Done!')

    return tsst, tab

def tf_synchrosqueeze(cwt_matr: np.ndarray, freqs: np.ndarray, ts: float,
                     scales: np.ndarray, deltaScales: np.ndarray, threshold: float,
                     numProc: int) -> Tuple[np.ndarray, np.ndarray]:

    sst, wab = freq_synchrosqueeze(cwt_matr, freqs, ts, scales, deltaScales, threshold, numProc)
    tsst, tab = time_synchrosqueeze(sst, freqs, ts, threshold, numProc)
    return tsst, wab

@njit(parallel=True, fastmath=True)
def _freq_agregate(deltaFreqs: np.ndarray, borderFreqs: np.ndarray,
                     aScale: np.ndarray, wab: np.ndarray, tr_matr: np.ndarray,
                     St: np.ndarray):
    for b in prange(St.shape[1]):        # Time
        for w in prange(St.shape[0]):    # Frequency
            components = np.logical_and(wab[:,b] > borderFreqs[w],
                                        wab[:,b] <= borderFreqs[w+1])

            St[w,b] = (tr_matr[components,b] * aScale[components]).sum() / deltaFreqs[w]


@njit(parallel=True, fastmath=True)
def _time_agregate(ts: float, time: np.ndarray, tab: np.ndarray,
                   tr_matr: np.ndarray, tsst: np.ndarray):
    for w in prange(tsst.shape[0]):        # Frequency
        for b in prange(tsst.shape[1]):    # Time
            components = np.logical_and(tab[w,:] > time[b],
                                        tab[w,:] <= time[b+1])
            tsst[w,b] = (tr_matr[w,components]).sum() / (2*np.pi)

def reconstruct(sst: np.ndarray, C_psi: complex,
                freqs: np.ndarray)-> np.ndarray:
    """Reconstruct signal from its SSWT

    Parameters
    ----------
    sst : np.ndarray
        SSWT Matrix
    wavelet : pywt.ContinuousWavelet
        Wavelet Family used in the SST

    Returns
    -------
    np.ndarray
        Reconstructed signal samples
    """
    deltaFreqs, _ = getDeltaAndBorderFreqs(freqs)
    signalR = (1/C_psi) * (sst * deltaFreqs[:,np.newaxis]).sum(axis=0)
    return signalR.real


def reconstructCWT(cwt: np.ndarray, wav: pywt.ContinuousWavelet,
                   scales: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    psi_w, x_w = wav.wavefun(wav.upper_bound-wav.lower_bound)
    C_w = abs(psi_w[np.argmin(np.abs(x_w))]) * (wav.upper_bound / 4)
    signalR = (1/C_w) *  np.sum(cwt / (scales[:, np.newaxis]**0.5)  * np.exp(-1j*freqs/scales)[:,np.newaxis], axis=0)
    # TODO: check scaling factor
    return signalR.real



if __name__=='__main__':

    import matplotlib.pyplot as plt

    from .configuration import Configuration
    from .utils import signal_utils as generator
    from .utils.measures_utils import renyi_entropy

    plt.close('all')
    logging.basicConfig(filename='sswt.log', filemode='w',
                        format='%(levelname)s - %(asctime)s - %(name)s:\n %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    stopTime = 5
    fs = 200
    signalLen = stopTime * fs

    t, ts = np.linspace(0, stopTime, signalLen, endpoint=False, retstep=True)

    # signal = generator.testSine(t, 0.2) + generator.testSine(t,1) + generator.testSine(t, 5) + generator.testSine(t,10)
    # f, signal = generator.testSig(t)
    # f, signal = generator.crossChrips(t, 2, 8, 2)
    # _, signal = generator.testChirp(t, 0.1, 30)
    # _, signal = generator.quadraticChirp(t, 1, 30)
    f, signal = generator.dualQuadraticChirps(t, (8,6),(2,3))
    # signal = np.zeros_like(t)
    # signal[fs:2*fs]=1.0

    wcf = 1
    wbw = 4

    maxFreq = 11
    minFreq = 0.1
    numFreqs = 20

    config = Configuration(
        minFreq=minFreq,
        maxFreq=maxFreq,
        numFreqs=numFreqs,
        ts=ts,
        wcf=wcf,
        wbw=wbw,
        waveletBounds=(-8,8),
        threshold=signal.max()/(100),
        plotFilt=False)

    wav = config.wav

    config.pad = 256

    scales, _, _, _ = calcScalesAndFreqs(ts, config.wcf, config.minFreq, config.maxFreq, config.numFreqs)

    sst, cwt, freqs, wab, tail = sswt(signal, **config.asdict())
    rentrCWT = renyi_entropy(cwt,2)
    rentrSST = renyi_entropy(sst,2)
    print(f'Rènyi entropy of CWT = {rentrCWT}')
    print(f'Rènyi entropy of SSWT = {rentrSST}')


    mainFig = plt.figure('Method comparison')
    gs = mainFig.add_gridspec(2, 2)
    mainAxes  = gs.subplots(sharex='col', sharey='row')
    mainFig.set_tight_layout(True)

    mainAxes[0,0].plot(t[:len(signal)], signal)
    mainAxes[0,0].set_title('Signal')


    mainAxes[1,0].pcolormesh(t, freqs, np.abs(cwt), cmap='viridis', shading='gouraud')
    mainAxes[1,0].set_title('Wavelet Transform')
    mainAxes[1,1].pcolormesh(t, freqs, np.abs(sst), cmap='viridis', shading='gouraud')
    mainAxes[1,1].set_title('Synchrosqueezing Transform')

    signalR_cwt = reconstructCWT(cwt, wav, scales, freqs)
    signalR_cwt /= signalR_cwt.max()
    signalR_sst = reconstruct(sst, config.C_psi, freqs)

    mainAxes[0, 1].plot(t, signal, label='Original', alpha=0.5)
    mainAxes[0, 1].plot(t, signalR_cwt, label='CWT', alpha=0.65)
    mainAxes[0, 1].plot(t, -1*signalR_sst, label='SST')
    mainAxes[0, 1].legend()
    mainAxes[0, 1].set_title('Reconstructed signal')

    #### Transforms comparison ####
    tsstFig, tsstAx = plt.subplots(1,4)

    # signal = np.zeros_like(t)
    # pulse_width = 10
    # pulse_start, pulse_stop = int(len(t)//5), int(len(t)//5) + pulse_width
    # signal[pulse_start:pulse_stop]=1

    sst, cwt, freqs, tab, tail = sswt(signal, **config.asdict())
    tsstAx[0].pcolormesh(t, freqs, np.abs(cwt), cmap='viridis', shading='gouraud')
    tsstAx[0].set_title('Wavelet Transform')
    tsstAx[1].pcolormesh(t, freqs, np.abs(sst), cmap='viridis', shading='gouraud')
    tsstAx[1].set_title('Synchrosqueezing Transform')

    config.transform = 'tsst'
    tsst, cwt, freqs, tab, tail = sswt(signal, **config.asdict())
    tsstAx[2].pcolormesh(t, freqs, np.abs(tsst), cmap='viridis', shading='gouraud')
    tsstAx[2].set_title('Time synchrosqueezing Transform')

    config.transform = 'tfr'
    tfr, cwt, freqs, tab, tail = sswt(signal, **config.asdict())
    tsstAx[3].pcolormesh(t, freqs, np.abs(tfr), cmap='viridis', shading='gouraud')
    tsstAx[3].set_title('Time Frequency Reasignment')

    print(f'Max values: CWT={abs(cwt).max()}, SST={abs(sst).max()}, TSST={abs(tsst).max()}, TFR={abs(tfr).max()}\n')

    #### Timing ####
    import timeit
    passes = 2
    config.pad = 0
    time = timeit.timeit("sswt(signal, **config.asdict())", globals=globals(), number=passes)
    print(f'Excecution time for {passes} passes and {config.numProc} processes = {time}s')
    print(f'Execution time per signal second = {time / stopTime / passes} s/s')

    plt.show()
