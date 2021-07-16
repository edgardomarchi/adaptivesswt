#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger()

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

import pywt
from typing import Tuple

from configuration import Configuration

import signal_utils as generator
from freq_utils import getScale, calcScalesAndFreqs
from sswt import sswt


def calcScales(ts:float, wcf:float, nvPerBand:list, flo:float, fhi:float,
               log: bool=False, freqBands:list=None) -> np.ndarray:
    """Calcula las escalas a utilizar en la WT

    Parameters
    ----------
    ts : float
        Período de muestreo
    wcf : float
        Frecuencia central de la wavelet en relación a 'fs'
    nvPerBand : list
        Número de voces por banda
    flo : float
        Frecuencia mínima
    fhi : float
        Frecuencia máxima
    log : bool
        Determina si la distribución de frecuencias es logarítmica o lineal, por default True
    freqBands : list, optional
        Frecuencias límite entre bandas, by default None

    Returns
    -------
    np.ndarray
        Escalas para utilizar en la CWT de menor a mayor (i.e. de (fhi a flo])
    """
    if freqBands is None:
        if log:
            freqs = np.logspace(np.log10(flo), np.log10(fhi),
                                len(nvPerBand)+1, endpoint=True)
        else:
            freqs = np.linspace(flo, fhi, len(nvPerBand)+1, endpoint=True)
    else:
        assert len(freqBands) == len(nvPerBand)+1,\
            "Error: La especificación de bandas y numero de voces no es coherente!"
        freqs = freqBands
    scales = np.zeros(np.array(nvPerBand).sum())
    nv = 0
    for i in range(len(nvPerBand)):
        bScales, _, _, _ = calcScalesAndFreqs(ts, wcf, freqs[i], freqs[i+1], nvPerBand[i], log)
        scales[nv:nv+nvPerBand[i]] = bScales
        nv += nvPerBand[i]
    return scales


def calcBandScales(fs:float, wcf:float, numFreqs:int,
                   flo:float, fhi:float, log:bool=False) -> np.ndarray:
    """Calcula las escalas para la WT dentro de una banda determinada.

    Parameters
    ----------
    fs : float
        Frecuencia de muestreo
    wcf : float
        Frecuencia central de la wavelet en realción a 'fs'
    numFreqs : int
        Cantidad de frecuencias dentro de la banda
    flo : float
        Frecuencia mínima
    fhi : float
        Frecuencia máxima
    log : bool
        True para separación logaritmica de frecuencias

    Returns
    -------
    np.ndarray
        Escalas para la banda de menor a mayor (i.e. de (fhi a flo])
    """
    minScale = getScale(fhi, 1/fs, wcf)
    maxScale = getScale(flo, 1/fs, wcf)
    if log:
        return np.logspace(np.log10(minScale), np.log10(maxScale), numFreqs,
                           endpoint=False)
    return np.linspace(minScale, maxScale, numFreqs, endpoint=False)


def proportional(nv: int, spectrum: np.ndarray) -> np.ndarray:
    """assign n seats proportionaly to votes using Hagenbach-Bischoff quota
    :param nseats: int number of seats to assign
    :param votes: iterable of int or float weighting each party
    :result: list of ints seats allocated to each party
    """
    quota = sum(spectrum) / (1 + nv)
    frac = np.array([freq/quota for freq in spectrum])
    res = np.array([int(f) for f in frac])
    n = nv - res.sum() # number of wavelets remaining to allocate
    if n==0: return res #done
    if n<0: return [min(x,nv) for x in res] # see siamii's comment
    # give the remaining wavelets to the n frequencies with the largest remainder
    remainders = [ai-bi for ai,bi in zip(frac,res)]
    limit=sorted(remainders,reverse=True)[n-1]
    # n frequencies with remainter larger than limit get an extra wavelet
    for i,r in enumerate(remainders):
        if r >= limit:
            res[i] += 1
            n -= 1 # attempt to handle perfect equality
            if n==0: return res #done
    assert False #should never happen


def calcNumWavelets(spectrum:np.ndarray, freqs:np.ndarray,
                    plotBands:bool=False) -> np.ndarray:
    """Calcula la cantidad de wavelets por banda en función del espectro

    Parameters
    ----------
    spectrum : np.ndarray
        Espectro de la señal a analizar
    freqs : np.ndarray
        Eje de frecuencias
    plotBands : bool
        Indica si plotear el espectro y bandas asignadas

    Returns
    -------
    np.ndarray
        Cantidad de wavelets a asignar por banda
    """
    numWavelets = proportional(len(spectrum), spectrum)
    rem = len(spectrum) - numWavelets.sum()
    logger.debug('Frecuencias centrales:\n %s', freqs)
    logger.debug('Resto: %d', rem)

    if plotBands:
        fig, axes = plt.subplots(2,1,sharex=True)
        axes[0].stem(freqs, abs(spectrum))
        axes[0].set_title('Potencia')
        axes[1].stem(freqs, numWavelets)
        axes[1].set_title('Cantidad de wavelets por banda')
        fig.suptitle('Espectro y frecuencias asignadas por banda')

    return numWavelets


def adaptive_sswt(iters:int=2, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Realiza `iters` calculos de la SSWT adaptando las escalas a la energía por banda de la señal

    Parameters
    ----------
    iters : int, optional
        Cantidad de iteraciones de adaptación, by default 2
    **kwargs: dict
        Mismos argumentos que `sswt()`

    Returns
    -------
    tuple
        (St:np.ndarray, Wt:np.ndarray, freqs:np.ndarray, scales:np.ndarray) Matrices con las transformadas ST y WT\
            y vectores con frecuencias y escalas de análisis.
    """
    minimumFreq = 1 / (kwargs['ts'] * len(args[0]))
    maximumFreq = 1 / (kwargs['ts'] * 2)
    logger.debug("---- Mínima frecuencia para esta señal: %s", minimumFreq)
    # Transformada inicial:
    sst, cwt, freqs, scales_adp = sswt(*args, **kwargs)

    for i in range(iters):
        logger.debug("******************* Iteración: %d ********************", i)
        spectrum = abs(sst).sum(axis=1)
        #del sst
        #del cwt
        numWavelets = calcNumWavelets(spectrum, freqs, plotBands=True)

        logger.debug("---- Número de frecuncias asignadas por banda:\n%s\n----------", numWavelets)
        logger.debug("---- Frecuencias centrales de análisis:\n%s\n----------", freqs)
        limits = (freqs[:-1] + freqs[1:]) / 2
        logger.debug("---- Límites entre bandas de análisis:\n%s\n----------", limits)

        freqBands = np.concatenate((np.array([2*freqs[0]-limits[0]]),#np.array([kwargs['maxFreq']]), # 
                                    limits,#[::-1],
                                    np.array([2*freqs[-1]-limits[-1]]))) #np.array([kwargs['minFreq']]))) # 
        # Si la segunda frecuencia está muy alejada de la primera el límite entre ellas
        # puede ser más grande que 2 veces la primer frecuencia.
        freqBands = np.where(freqBands >= minimumFreq, freqBands, minimumFreq)
        freqBands = np.where(freqBands <= maximumFreq, freqBands, maximumFreq)

        logger.debug("---- Frecuncias límites entre bandas incluyendo extremos:\n %s\n----------", freqBands)

        scales_adp = calcScales(kwargs['ts'], kwargs['wcf'], numWavelets,
                                kwargs['minFreq'], kwargs['maxFreq'],
                                freqBands=freqBands, log=kwargs['log'])

        logger.debug("---- Escalas a utilizar para CWT:\n%s\n----------", scales_adp)
        kwargs['custom_scales'] = scales_adp

        #del freqs
        #gc.collect()
        sst, cwt, freqs, scales_adp = sswt(*args, **kwargs)
        if not sst.any():
            logger.warning('ATENCIÓN SSWT contiene sólo 0s')

        
    return sst, cwt, freqs, scales_adp

    
#%%
if __name__=='__main__':
    import scipy.signal as sp
    plt.close('all')
    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
#%% Parámetros

    stopTime = 15
    fs = 200
    signalLen = stopTime * fs    #2000

    t, ts = np.linspace(0, stopTime, signalLen, endpoint=False, retstep=True)



    mainFig = plt.figure('Comparación de métodos')
    gs = mainFig.add_gridspec(2, 3)
    mainAxes  = gs.subplots(sharex='col', sharey='row')
    mainFig.set_tight_layout(True)
    
#%% Señal de prueba



    #sig = pulsePosModSig(t,pw=t[1]*3, prf=1)
    #sig = hbSig(t)
    
    f = np.pi
    # _, sig = testChirp(t, 1, 40)
    # _, sig = testUpDownChirp(t,1,10)
    # sig = quadraticChirp(t, 1, 10)

    # sig=testSig(t)
    
    # sig=testSine(t,f)
    
    sig = generator.delta(t, 2)

    mainAxes[0,0].plot(t[:len(sig)], sig)
    mainAxes[0,0].set_title('Señal')

#%% Transformada
    wcf = 1#0.5
    wbw = 8

    maxFreq = fs/3
    minFreq = 0.1
    numFreqs = 200

    config = Configuration(
        minFreq=minFreq,
        maxFreq=maxFreq,
        numFreqs=numFreqs,
        ts=ts,
        wcf=wcf,
        wbw=wbw,
        waveletBounds=(-8,8),
        umbral=sig.max()/(100),
        numProc=1,
        log=False)

    # sig=sp.hilbert(sig)
    sst, cwt, freqs, scales = adaptive_sswt(0, sig, **config.asdict())
    #minFreq, maxFreq, numFreqs, ts=ts, wcf=wcf, wbw=wbw,
    #               umbral=sig.max()/100, numProc=2)
 
    fig = plt.figure("SSWT")
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(t, freqs)
    ax.plot_surface(X, Y, abs(sst))

    mainAxes[1,0].pcolormesh(t, freqs, np.abs(cwt),
                  cmap='viridis', shading='gouraud')
    #mainAxes[1,0].set_yscale('log')
    mainAxes[1,0].set_title('Wavelet Transform')
    mainAxes[1,1].pcolormesh(t, freqs, np.abs(sst),
                  cmap='viridis', shading='gouraud')
    mainAxes[1,1].set_title('Synchrosqueezing Transform')


    wav = pywt.ContinuousWavelet(f'cmor{config.wbw}-{config.wcf}')
    # wav = pywt.ContinuousWavelet(f'shan{config.wbw}-{config.wcf}')
    # wav = pywt.ContinuousWavelet(f'cgau4')
    # wav = pywt.ContinuousWavelet(f'fbsp4-{config.wbw}-{config.wcf}')
    wav.lower_bound, wav.upper_bound = config.waveletBounds
    psi, x = pywt.integrate_wavelet(wav)
    print(f'\n\nLongitud de PSI: {len(psi)}\n\n')
    C = np.pi * np.conjugate(psi[np.argmin(np.abs(x))])

    psi_w, x_w = wav.wavefun(wav.upper_bound-wav.lower_bound)
    C_w = abs(psi_w[np.argmin(np.abs(x_w))]) * (wav.upper_bound / 4)

    deltaFreqs = np.diff(freqs, append=(2*freqs[-1] - freqs[-2]))
    indexes  = np.where(np.logical_and(freqs >= minFreq, freqs<=maxFreq))[0]
    signalR = (1/C_w) *  np.sum(cwt / scales[:, np.newaxis] / 2, axis=0) # (1/C) * (sst[indexes,:]).sum(axis=0) #
    signalRfull = (1/C) * sst.sum(axis=0)
    
    mainAxes[0, 1].plot(t, sig, label='Señal', alpha=0.8)
    mainAxes[0, 1].plot(t, signalRfull.real, label='Full')
    mainAxes[0, 1].legend()
    mainAxes[0, 1].set_title('Señal reconstruída')

    iterations = 0
    iters = iterations + 1

    mse = np.zeros(iters)
    fig, specAx = plt.subplots(1)
    fig.suptitle('Espectros synchrosqueezed')
    spectrum = abs(sst.sum(axis=1))
    specAx.plot(freqs, spectrum, label='0 Iteraciones')

    # colors = ['b','g','r','o','c','m']
    # plotFilt = False
    # for it in range(iters):
        # print('****************************************************************************************************')
        # print(f'******************************************* Iteración: {it} *******************************************')
        # if it == iters-1:
    config.plotFilt = True
    sst, _, freqs, scales = adaptive_sswt(iters, sig, **config.asdict())
    spectrum = abs(sst.sum(axis=1))
    specAx.plot(freqs, spectrum, label=f'{iters} Iteraciones')
        # estFreq = freqs[spectrum.argmax()]
        # mse[it] = (estFreq - f)**2
        # logger.info('Frecuencia estimada: %s', estFreq)
    specAx.legend()
    

    mainAxes[1, 2].pcolormesh(t, freqs, np.abs(sst),
                  cmap='viridis', shading='gouraud')
    mainAxes[1, 2].set_title(f'Synchrosqueezing Transform luego de {iters} iteraciones')

    fig = plt.figure(f'SSWT adaptiva ({iters} iters)')
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(t, freqs)
    ax.plot_surface(X, Y, abs(sst))

    deltaFreqs = np.diff(freqs, append=(2*freqs[-1] - freqs[-2]))
    indexes  = np.where(np.logical_and(freqs >= minFreq, freqs<=maxFreq))[0]
    signalR = (1/C) * (sst[indexes,:]).sum(axis=0)
    
    mainAxes[0, 2].plot(t, signalR.real)
    mainAxes[0, 2].set_title('señal reconstruída con algorimto adaptivo')

    fig, ax = plt.subplots(1)
    ax.plot(freqs, np.abs(sst).sum(axis=1))
    fig.suptitle('Espectro de la SST')

    density = 1 - 0.75*sp.gaussian(int(len(signalR.real)*2.3), std=(len(signalR.real)/4))

    print(f'Tiempo de muestreo = {config.ts}')
    fig, ax = plt.subplots(1)
    ax.plot(np.fft.rfftfreq(len(signalR.real), d=config.ts), abs(np.fft.rfft(signalR.real)), label='Espectro Señal reconstruída')
    ax.plot(t, density[:len(signalR.real)], label='Función densidad')
    ax.legend()
    fig.suptitle('Comparación señal con función densidad')

    # fig, ax = plt.subplots(1)
    # ax.plot(range(iters), mse)
    # fig.suptitle('MSE en función de Iteraciones')
    
    plt.show()