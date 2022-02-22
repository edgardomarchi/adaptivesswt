#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger()

import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import pywt

import multiprocessing as mp
from numba import njit
from typing import Tuple

from .freq_utils import calcScalesAndFreqs, getDeltaAndBorderFreqs, getDeltaScales ,calcFilterLength
from .plot_utils import plotFilters

def sswt(signal: np.ndarray,
         minFreq: float,
         maxFreq: float,
         numFreqs: int,
         ts: float=1,
         threshold: float = 1,
         wav: pywt.ContinuousWavelet=pywt.ContinuousWavelet('cmor1-0.5'),
         custom_scales: np.ndarray=None,
         log: bool=False,
         pad: bool=False,
         numProc: int = 4,
         plotFilt: bool=True,
         C_psi: complex = None,
         **kwargs
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    log : bool, optional
        If True a logarithmic distribution of frequencies is used, linear if False, by default False
    pad : bool, optional
        If True adds padding for overlap-and-add applications, by default False
    plotFilt : bool, optional
        Wavelet filters spectrum is ploted if True (affects performance), by default False
    C_psi: complex, optional
        Reconstruction coeficient. Needed if pad is True.

    Returns
    -------
    tuple
        (St:np.ndarray, Wt:np.ndarray, freqs:np.ndarray, scales:np.ndaray, tail:np.ndarray)
        Matrices with SST, WT, analysis frequencies, scales used and signal tail for overlap (if pad is False tail is None).
    """
    
    #####################
    #%% Wavelet Transform
    #####################
    wcf = wav.center_frequency

    #### Scales ####
    if custom_scales is None:
        scales, freqs, deltaScales, deltaFreqs = calcScalesAndFreqs(ts, wcf, minFreq, maxFreq, numFreqs, log)
    else:
        scales = custom_scales
        logger.debug('Using custom scales')
        
        deltaScales = getDeltaScales(scales)
        # deltaScales = -1*np.diff(scales, prepend=scales[1]) # last delta is equal to the previous
        # deltaScales[0]*=-1                                  #  "
    
    logger.debug('Scales: \n%s\n', scales)
    logger.debug('Delta scales: \n%s\n', deltaScales)

    maxWavLen = 0 
    # padding for streaming:
    if pad:
        maxWavLen = calcFilterLength(wav, scales.max())
        signal = np.pad(signal, (0, maxWavLen), mode='constant')
        print(f'Max wav len = {maxWavLen}')     

    #### CWT ####
    if plotFilt: plotFilters(wav, scales, ts, signal)
    cwt_matr, freqs = pywt.cwt(signal, scales, wav, sampling_period=ts,
                               method='fft')

    #### SSWT ####
    numbaParallel = kwargs.get('numbaParallel', False)
    St = synchrosqueeze(cwt_matr, freqs, ts, scales, deltaScales, log, threshold, numProc, numbaParallel)
    if pad:
        assert C_psi is not None, 'Atention: C_psi is needed if pad is True'
        tail = reconstruct(St[:,-maxWavLen:], C_psi, freqs)
        lastIdx = -maxWavLen
    else:
        tail, lastIdx = None, None

    return St[:,:lastIdx], cwt_matr[:,:lastIdx], freqs, scales, tail


def synchrosqueeze(cwt_matr: np.ndarray, freqs: np.ndarray, ts: float, scales: np.ndarray,
                   deltaScales: np.ndarray, log: bool, threshold: float,
                   numProc: int, numbaParallel: bool = False):

    scaleExp = -3/2
    aScale = (scales ** scaleExp) * deltaScales
    logger.debug('a_k^{%s} * da_k: \n%s\n', scaleExp, aScale)

    #### Frecuencies ####
    deltaFreqs, borderFreqs = getDeltaAndBorderFreqs(freqs)
    
    logger.debug("CWT frequencies: \n%s\n", freqs)
    logger.debug("deltaFreqs: \n%s\n", deltaFreqs)
    logger.debug("Frequency band limits: \n%s\n", borderFreqs)

    #%% Map (a,b) -> (w(a,b), b)
   
    logger.info('Calculating instantaneous frequencies...')
    # Eq. (13) - "The Synchrosqueezing algorithm for time-varying spectral
    # analysis: robustness properties and new paleoclimate applications" - 
    # G. Thakur, E. Brevdo, N. S. Fučkar, and Hau-Tieng Wu:
    #
    # dCWT = np.gradient(cwt_matr, axis=1)
    # wab = np.zeros_like(cwt_matr, dtype='float64')
    # pos = abs(cwt_matr) > threshold
    # wab[pos] = np.imag(dCWT[pos] / cwt_matr[pos]) / (2 * np.pi * ts)

    # Eq. (7) - "Adaptive synchrosqueezing based on a quilted short time
    # Fourier transform" - A. Berrian, N. Saito:
    #
    cwt_matr_p = np.roll(cwt_matr,-1)
    cwt_matr_p[:,-1] = 0
    wab = np.angle(np.divide(cwt_matr_p, cwt_matr,
                             out=np.zeros_like(cwt_matr),
                             where=abs(cwt_matr)>threshold)) / (2 * np.pi * ts)
    # Last term is added in order to convert from normalized omega to frecuency in Hz
   
    St = np.zeros_like(cwt_matr)
    
    ####################################
    #%% Sychrosqueezing parallel process
    ####################################

    if numbaParallel:
        _freqSearchNumbaParallel(deltaFreqs, borderFreqs, aScale, wab, cwt_matr, St)
    else:
        jobs = mp.JoinableQueue()
        results = mp.Queue()

        processes = create_processes(deltaFreqs, borderFreqs, aScale, jobs, results, numProc)
        chunkSize = add_jobs(cwt_matr, wab, numProc, jobs)
    
        try:
            jobs.join()
        except KeyboardInterrupt: # May not work on Windows
            logger.info('... canceling synchrosqueezing process...')
        while not results.empty(): # Safe because all jobs have finished
            job, StChunk = results.get()  # _nowait()?
            St[:,job*chunkSize:(job+1)*chunkSize] = StChunk[:,:]
        for process in processes:
            process.terminate()
    logger.info('Synchrosqueezing Done!')

    return St


def create_processes(deltaFreqs, borderFreqs, aScale,
                     jobs, results, concurrency):
    processes = []
    for i in range(concurrency):
        processes.append(mp.Process(target=freqMap,
                         args=(deltaFreqs, borderFreqs, aScale, jobs, results)))
        processes[i].daemon = True
        processes[i].start()
    logger.info('Created %s processes.\n', concurrency)
    return processes


def add_jobs(cwtmatr, wab, numJobs, jobs):
    chunkSize = int(cwtmatr.shape[1]//numJobs)
    for job in range(numJobs):
        cwtmatrChunk, wabChunk = (cwtmatr[:,job*chunkSize:(job+1)*chunkSize],
                                  wab[:,job*chunkSize:(job+1)*chunkSize])
        jobs.put((job, cwtmatrChunk, wabChunk))
    logger.info('Queued %s jobs.\n', numJobs)
    return chunkSize


def freqMap(deltaFreqs: np.ndarray, borderFreqs: np.ndarray, aScale: np.ndarray,
            jobs: mp.JoinableQueue, results: mp.Queue):
    
    while True:
        job, tr_matr, wab = jobs.get()
        St = np.zeros_like(tr_matr)
    
        _freqSearch(deltaFreqs, borderFreqs, aScale, wab, tr_matr, St)
            
        results.put((job, St))
        jobs.task_done()


@njit(fastmath=True)
def _freqSearch(deltaFreqs: np.ndarray, borderFreqs: np.ndarray,
                aScale: np.ndarray, wab: np.ndarray, tr_matr: np.ndarray,
                St: np.ndarray):
    
    for b in range(St.shape[1]):        # Time
        for w in range(St.shape[0]):    # Frequency
            components = np.logical_and(wab[:,b] > borderFreqs[w],
                                        wab[:,b] <= borderFreqs[w+1])
                
            St[w,b] = (tr_matr[components,b] * aScale[components]).sum() / deltaFreqs[w]

@njit(parallel=True, fastmath=True)
def _freqSearchNumbaParallel(deltaFreqs: np.ndarray, borderFreqs: np.ndarray,
                     aScale: np.ndarray, wab: np.ndarray, tr_matr: np.ndarray,
                     St: np.ndarray):
    for b in range(St.shape[1]):        # Time
        for w in range(St.shape[0]):    # Frequency
            components = np.logical_and(wab[:,b] > borderFreqs[w],
                                        wab[:,b] <= borderFreqs[w+1])
                
            St[w,b] = (tr_matr[components,b] * aScale[components]).sum() / deltaFreqs[w]

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
    return signalR.real


if __name__=='__main__':
    from configuration import Configuration
    import signal_utils as generator

    plt.close('all')
    logging.basicConfig(filename='sswt.log', filemode='w',
                        format='%(levelname)s - %(asctime)s - %(name)s:\n %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
        
    stopTime = 12
    fs = 2000
    signalLen = stopTime * fs

    t, ts = np.linspace(0, stopTime, signalLen, endpoint=False, retstep=True)

    # signal = generator.testSine(t, 0.2) + generator.testSine(t,1) + generator.testSine(t, 5) + generator.testSine(t,10)
    signal = generator.testSig(t)
    # _, signal = generator.testChirp(t, 0.1, 30)
    # _, signal = generator.quadraticChirp(t, 1, 30)
    # signal = np.zeros_like(t)
    # signal[fs:2*fs]=1.0

    wcf = 0.5
    wbw = 2

    maxFreq = 10
    minFreq = 0.1
    numFreqs = 128

    assert fs/len(signal)<= minFreq, 'ATETENTION: Minimum analysis frecuequency is lower than L/N!'
    print(f'Wav len pre calculated = {fs/minFreq}')

    config = Configuration(
        minFreq=minFreq,
        maxFreq=maxFreq,
        numFreqs=numFreqs,
        ts=ts,
        wcf=wcf,
        wbw=wbw,
        waveletBounds=(-3,3),
        threshold=signal.max()/(100),
        numProc=4,
        log=False,
        plotFilt=False)

    wav = config.wav  # pywt.ContinuousWavelet(f'cmor{config.wbw}-{config.wcf}')
    #wav.lower_bound, wav.upper_bound = config.waveletBounds

    config.pad = True
    sst, cwt, freqs, scales, tail = sswt(signal, **config.asdict())

    # fig = plt.figure("CWT")
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(t, freqs)
    # ax.plot_surface(X, Y, abs(cwt), cmap='viridis')

    # fig = plt.figure("SSWT")
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(t, freqs)
    # ax.plot_surface(X, Y, abs(sst), cmap='viridis')

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
    signalR_sst = reconstruct(sst, config.C_psi, freqs)

    mainAxes[0, 1].plot(t, signal, label='Original', alpha=0.5)
    #mainAxes[0, 1].plot(t, signalR_cwt, label='CWT', alpha=0.65)
    mainAxes[0, 1].plot(t, -1*signalR_sst, label='SST')
    mainAxes[0, 1].legend()
    mainAxes[0, 1].set_title('Reconstructed signal')

    import timeit
    passes = 2
    config.pad = False
    time = timeit.timeit("sswt(signal, **config.asdict())", globals=globals(), number=passes)
    print(f'Excecution time for {passes} passes and {config.numProc} processes = {time}s')
    print(f'Execution time per signal second = {time / stopTime / passes} s/s')

    #error = (signal - signalR_sst)**2
    # fig, ax = plt.subplots(1)
    # ax.plot(t, error)
    # fig.suptitle('Error cuadrático')
    #print(f'MSE: {error.mean()}')

    plt.show()