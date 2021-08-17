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

from freq_utils import calcScalesAndFreqs, getDeltaAndBorderFreqs
from plot_utils import plotFilters

def sswt(signal: np.ndarray,
         minFreq: float,
         maxFreq: float,
         numFreqs: int,
         ts: float=1,
         wcf: float=1,
         wbw: float=1.5,
         waveletBounds=(-8,8),
         umbral: float = 1,
         numProc: int = 4,
         custom_scales: np.ndarray=None,
         log: bool=True,
         plotFilt: bool=True
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcula la transformada Synchrosqueezed Wavelet.

    Parameters
    ----------
    signal : np.ndarray
        Señal a transformar.
    minFreq : float
        Mínima frecuencia de análisis (min = Fs/N).
    maxFreq : float
        Máxima frecuencia de análisis (max = Fs/2).
    numFreqs : float
        Número de frecuencias de análisis.
    ts : float, optional
        Tiempo de muestreo, by default 1
    wcf : float, optional
        Frecuencia central de la wavelet relativa a fs, by default 1
    wbw : float, optional
        Ancho de banda de la wavelet relativa a la frecuencia central, by default 1.5
    umbral : float, optional
        Umbral de detección de la transformada, by default 1
    numProc : int, optional
        Numero de sub-procesos a utilizar, by default 4
    custom_scales : np.ndarray, optional
        Vector de escalas personalizadas para la WT, by default None
    log : bool, optional
        Indica si se utlizan frecuencias separadas logarítmicamente, by default True
    plotFilt : bool, optional
        Flag que indica si imprimir el espectro junto con los filtros de las wavelets, by default True

    Returns
    -------
    tuple
        (St:np.ndarray, Wt:np.ndarray, freqs:np.ndarray) Matrices con las transformadas ST y WT y vector con frecuencias de análisis.
    """
    #%% Transformada Wavelet

    #### Escalas ####
    if custom_scales is None:
        scales, freqs, deltaScales, deltaFreqs = calcScalesAndFreqs(ts, wcf, minFreq, maxFreq, numFreqs, log)
    else:
        scales = custom_scales
        logger.debug('Usando escalas custom')
        
        deltaScales = -1*np.diff(scales, prepend=scales[1]) # el último da es igual al anterior
        deltaScales[0]*=-1                                 #  "
    
    logger.debug('Escalas: \n%s\n', scales)
    logger.debug('Delta escalas: \n%s\n', deltaScales)

    scaleExp = -3/2  # LOG: ** (-1/2) - LIN: **(-3/2)
    aScale = (scales ** scaleExp) * deltaScales

    logger.debug('a_k^{%s} * da_k: \n%s\n', scaleExp, aScale)

    wav = pywt.ContinuousWavelet(f'cmor{wbw}-{wcf}')   
    # wav = pywt.ContinuousWavelet(f'shan{config.wbw}-{config.wcf}')
    # wav = pywt.ContinuousWavelet(f'cgau4')
    # wav = pywt.ContinuousWavelet(f'fbsp4-{wbw}-{wcf}')
    wav.lower_bound, wav.upper_bound = waveletBounds

    #### CWT ####
    if plotFilt: plotFilters(wav, scales, ts, signal)

    cwt_matr, freqs = pywt.cwt(signal, scales, wav, sampling_period=ts,
                               method='fft')
    
    #### Frecuencias ####
    # if custom_scales is not None:
    #     deltaFreqs = np.diff(freqs, prepend=freqs[1]) # append=(2*freqs[-1] - freqs[-2]))
    #     deltaFreqs[0] *= -1
    
    # borderFreqs = np.concatenate((freqs-deltaFreqs/2,
    #                               np.array([freqs[-1]+deltaFreqs[-1]/2])))
    
    deltaFreqs, borderFreqs = getDeltaAndBorderFreqs(freqs)
    
    # waveletBandwidths = ((wcf / ts) / wbw) / scales

    # deltaFreqs = np.where(deltaFreqs < waveletBandwidths, deltaFreqs, waveletBandwidths)

    # fig, ax = plt.subplots(1)
    # ax.plot(freqs, deltaFreqs, label='df')
    # ax.plot(freqs, deltaScales, label='da')
    # ax.plot(freqs, aScale, label='Coeficiente')
    # ax.legend()
    # fig.suptitle('Factores de escala')

    logger.debug("Frecuencias de la CWT: \n%s\n", freqs)
    logger.debug("deltaFreqs: \n%s\n", deltaFreqs)
    logger.debug("Límites entre bandas: \n%s\n", borderFreqs)

    #%% Mapeo (a,b) -> (w(a,b), b)
   
    logger.info('Calculando frecuencias instantáneas...')
    # Eq. (13) - "The Synchrosqueezing algorithm for time-varying spectral
    # analysis: robustness properties and new paleoclimate applications" - 
    # G. Thakur, E. Brevdo, N. S. Fučkar, and Hau-Tieng Wu
    # dCWT = np.gradient(cwt_matr, axis=1)
    # wab = np.zeros_like(cwt_matr, dtype='float64')
    # pos = abs(cwt_matr) > umbral
    # wab[pos] = np.imag(dCWT[pos] / cwt_matr[pos]) / (2 * np.pi * ts)

    # Eq. (7) - "Adaptive synchrosqueezing based on a quilted short time
    # Fourier transform" - A. Berrian, N. Saito
    cwt_matr_p = np.roll(cwt_matr,-1)
    cwt_matr_p[:,-1] = 0
    wab = np.angle(np.divide(cwt_matr_p, cwt_matr,
                             out=np.zeros_like(cwt_matr),
                             where=abs(cwt_matr)>umbral)) / (2 * np.pi * ts)
    # El último término se agrega para pasar de omega normalizado a frecuencia

    #%% Transformada Sychrosqueezing
    jobs = mp.JoinableQueue()
    results = mp.Queue()

    # widths = 2 * (wcf/wbw) / scales / ts
    # print(f'Anchos de las wavelets: {widths}')

    # windows = np.identity(numFreqs)
    # fig = plt.figure('Envolventes de las wavelets')
    # ax = fig.add_subplot(111, label='Wavelets')
    # for i in range(numFreqs):
    #     windows[i,:] = np.convolve(windows[i,:],
    #                                sp.windows.gaussian(numFreqs-1,  # 1 / np.sqrt(np.pi * wbw) * 
    #                                                    numFreqs / (scales[i] * np.sqrt(2 / (wbw)))),
    #                                mode='same')
    #     ax.plot(windows[i,:])
    
    # density = windows.sum(axis=1)
    # da = np.diff(density, prepend=density[0])
    
    # print(f'Density wavelets: {density}')
    
    # plt.figure('Densidad')
    # plt.plot(density)

    #aScale *= density
    #aScale *= da

    create_processes(freqs, deltaFreqs, borderFreqs, 
                     scales, deltaScales, aScale,
                     jobs, results, numProc)
    
    chunkSize = add_jobs(cwt_matr, wab, numProc, jobs)
    
    St = np.zeros_like(cwt_matr)
 
    try:
        jobs.join()
    except KeyboardInterrupt: # May not work on Windows
        print('... canceling...')
    while not results.empty(): # Safe because all jobs have finished
        job, StChunk = results.get_nowait()
        St[:,job*chunkSize:(job+1)*chunkSize] = StChunk[:,:]
    print('Synchrosqueezing Done!')
    return St, cwt_matr, freqs, scales


def create_processes(freqs, deltaFreqs, borderFreqs,
                     scales, deltaScales, aScale, jobs, results, concurrency):
    for _ in range(concurrency):
        process = mp.Process(target=freqMap,
                             args=(freqs, deltaFreqs, borderFreqs, aScale, jobs, results))
        process.daemon = True
        process.start()
    logger.info(f'{concurrency} procesos creados.\n')


def add_jobs(cwtmatr, wab, numJobs, jobs):
    chunkSize = int(cwtmatr.shape[1]//numJobs)
    for job in range(numJobs):
        cwtmatrChunk, wabChunk = (cwtmatr[:,job*chunkSize:(job+1)*chunkSize],
                                  wab[:,job*chunkSize:(job+1)*chunkSize])
        jobs.put((job, cwtmatrChunk, wabChunk))
    logger.info(f'{numJobs} trabajos acolados.\n')
    return chunkSize


def freqMap(freqs: np.ndarray, deltaFreqs: np.ndarray, borderFreqs: np.ndarray,
            aScale: np.ndarray, jobs: mp.JoinableQueue, results: mp.Queue):
    
    while True:
        job, tr_matr, wab = jobs.get()
        St = np.zeros_like(tr_matr)
    
        St = _freqSearch(deltaFreqs, borderFreqs, aScale, wab, tr_matr, St)
            
        results.put((job, St))
        jobs.task_done()


@njit
def _freqSearch(deltaFreqs: np.ndarray, borderFreqs: np.ndarray, aScale: np.ndarray, wab: np.ndarray,
                tr_matr: np.ndarray, St: np.ndarray) -> np.ndarray:

    
    for b in range(St.shape[1]):        # Tiempo
        for w in range(St.shape[0]):    # Frecuencia
            components = np.logical_and(wab[:,b] > borderFreqs[w],
                                        wab[:,b] <= borderFreqs[w+1])
                
            St[w,b] = (tr_matr[components,b] * aScale[components]).sum() / deltaFreqs[w]
    return St 

def reconstruct(sst: np.ndarray, wavelet: pywt.ContinuousWavelet, freqs: np.ndarray)-> np.ndarray:
    """Reconstruye la señal original a partir de su SSWT

    Parameters
    ----------
    sst : np.ndarray
        Matriz con la SSWT
    wavelet : pywt.ContinuousWavelet
        Familia de wavelets utilizada para la SSWT

    Returns
    -------
    np.ndarray
        Las muestras de la señal
    """
    psi, x = pywt.integrate_wavelet(wavelet)
    print(f'\n\nLongitud de PSI: {len(psi)}\n\n')
    C = np.pi * np.conjugate(psi[np.argmin(np.abs(x))])
    deltaFreqs, _ = getDeltaAndBorderFreqs(freqs)
    signalR = (1/C) * (sst * deltaFreqs[:,np.newaxis]).sum(axis=0)
    return signalR.real

def reconstructCWT(cwt: np.ndarray, wav: pywt.ContinuousWavelet, scales: np.ndarray, freqs: np.ndarray) -> np.ndarray:
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
    fs = 1000
    signalLen = stopTime * fs    #2000

    t, ts = np.linspace(0, stopTime, signalLen, endpoint=False, retstep=True)

    # signal = generator.testSine(t, 1) + generator.testSine(t,10) + generator.testSine(t, 20) + generator.testSine(t,30) + generator.testSine(t,39)
    # signal = generator.testSig(t)
    _, signal = generator.testChirp(t, 0.1, 40)
    # signal = generator.quadraticChirp(t, 1, 40)

    wcf = 1#0.5
    wbw = 4

    maxFreq = 50 #fs/3
    minFreq = 0.1
    numFreqs = 50

    assert fs/len(signal)<= minFreq, 'ATENCIÓN: La mínima frecuencia de análisis está por debajo de L/N!'

    config = Configuration(
        minFreq=minFreq,
        maxFreq=maxFreq,
        numFreqs=numFreqs,
        ts=ts,
        wcf=wcf,
        wbw=wbw,
        waveletBounds=(-8,8),
        umbral=signal.max()/(100),
        numProc=2,
        log=False,
        plotFilt=True)

    wav = pywt.ContinuousWavelet(f'cmor{config.wbw}-{config.wcf}')
    wav.lower_bound, wav.upper_bound = config.waveletBounds

    # sig=sp.hilbert(sig)
    sst, cwt, freqs, scales = sswt(signal, **config.asdict())

    from scipy import signal as sp
    cwtmatr = sp.cwt(signal, sp.morlet2, scales)

    fig = plt.figure("CWT SciPy")
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(t, freqs)
    ax.plot_surface(X, Y, abs(cwtmatr), cmap='viridis')

    fig = plt.figure("CWT PyWavelets")
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(t, freqs)
    ax.plot_surface(X, Y, abs(cwt), cmap='viridis')

    fig = plt.figure("SSWT")
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(t, freqs)
    ax.plot_surface(X, Y, abs(sst), cmap='viridis')

    deltaFreqs, _ = getDeltaAndBorderFreqs(freqs) 
    spec = abs(sst*deltaFreqs[:,np.newaxis]).sum(axis=1)
    sfig = plt.figure('Espectro Synchrosqueezed')
    sax = sfig.add_subplot(111)
    sax.plot(freqs, spec)

    mainFig = plt.figure('Comparación de métodos')
    gs = mainFig.add_gridspec(2, 2)
    mainAxes  = gs.subplots(sharex='col', sharey='row')
    mainFig.set_tight_layout(True)

    mainAxes[0,0].plot(t[:len(signal)], signal)
    mainAxes[0,0].set_title('Señal')

    mainAxes[1,0].pcolormesh(t, freqs, np.abs(cwt), cmap='viridis', shading='gouraud')
    mainAxes[1,0].set_title('Wavelet Transform')
    mainAxes[1,1].pcolormesh(t, freqs, np.abs(sst), cmap='viridis', shading='gouraud')
    mainAxes[1,1].set_title('Synchrosqueezing Transform')

    signalR_cwt = reconstructCWT(cwt, wav, scales, freqs)
    signalR_sst = reconstruct(sst, wav, freqs)

    mainAxes[0, 1].plot(t, signal, label='Original', alpha=0.5)
    # mainAxes[0, 1].plot(t, signalR_cwt, label='CWT', alpha=0.65)
    mainAxes[0, 1].plot(t, signalR_sst, label='SST')
    mainAxes[0, 1].legend()
    mainAxes[0, 1].set_title('Señal reconstruída')

    sax.plot(np.fft.rfftfreq(len(signalR_sst),ts), np.abs(np.fft.rfft(signalR_sst)), label='FFT reconstr.')
    sax.legend()

    error = (signal - signalR_sst)**2
    fig, ax = plt.subplots(1)
    ax.plot(t, error)
    fig.suptitle('Error cuadrático')

    print(f'MSE: {error.mean()}')

    plt.show()