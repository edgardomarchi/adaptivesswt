#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from multiprocessing import cpu_count
from typing import Optional, Tuple

import numpy as np
import pywt

from . import __backendConfig, __setBackend
from .utils.freq_utils import (
    calcFilterLength,
    calcScalesAndFreqs,
    getDeltaAndBorderFreqs,
    getDeltaScales,
)
from .utils.plot_utils import plot_cwt_filters

logger = logging.getLogger(__name__)

backendConfig: dict = __backendConfig

match backendConfig['backend']:
    case 'opencl':
        logger.info('Using "opencl" backend')
        try:
            from .sswtcl import _freq_agregate_cl, _freq_extract_cl, _time_agregate_cl
            _freq_agregate, _freq_extract, _time_agregate = _freq_agregate_cl, _freq_extract_cl, _time_agregate_cl
        except Exception as e:
            __setBackend('numba')
            logger.error('Could not use "opencl". Switching to "numba".')
            logger.error('Exception: %s', e)

    case _:
        logger.info('Using "numba" backend')
        from numba import set_num_threads

        from .sswtnb import _freq_agregate_nb, _freq_extract_nb, _time_agregate_nb
        _freq_agregate, _freq_extract, _time_agregate = _freq_agregate_nb, _freq_extract_nb, _time_agregate_nb


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
    if backendConfig['backend'] == 'numba': set_num_threads(num_processes)  # type: ignore  # set_num_threads is imported only if numba is used

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
    match transform:
        case 'set':
            sst = _freq_extract(deltaFreqs, borderFreqs, aScale, wab, cwt_matr, sst)
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

    tsst = _time_agregate(time, tab, cwt_matr, tsst)
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
    sst = np.zeros_like(cwt_matr)
    sst = _freq_agregate(deltaFreqs, borderFreqs, aScale, wab, cwt_matr, sst)
    tfr = np.zeros_like(cwt_matr)
    tfr = _time_agregate(time, tab, sst, tfr)

    return tfr, (wab, tab)

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
