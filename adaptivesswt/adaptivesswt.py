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
from sswt import sswt, reconstruct, reconstructCWT


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


def calcNumWavelets(spectrum:np.ndarray, freqs:np.ndarray, method='proportional', thrsh=1/20,
                    plotBands:bool=True) -> np.ndarray:
    """Calcula la cantidad de wavelets por banda en función del espectro

    Parameters
    ----------
    spectrum : np.ndarray
        Espectro de la señal a analizar
    freqs : np.ndarray
        Eje de frecuencias
    method : str
        Método de reasignación de frecuencias: 'proportional', 'threshold'
    plotBands : bool
        Indica si plotear el espectro y bandas asignadas

    Returns
    -------
    np.ndarray
        Cantidad de wavelets a asignar por banda
    """
    if method=='proportional':
        numWavelets = proportional(len(spectrum), spectrum)
    elif method=='threshold':
        numWavelets = np.where(spectrum > (spectrum.max() * thrsh), 1, 0)
        numUnusedWavelets = len(spectrum) - numWavelets.sum()  
        numWavelets += proportional(numUnusedWavelets, spectrum)
    else:  # For now, proportional is the default
        numWavelets = proportional(len(spectrum), spectrum)

    rem = len(spectrum) - numWavelets.sum()
    logger.debug('Frecuencias centrales:\n %s', freqs)
    logger.debug('Resto: %d', rem)

    if plotBands:
        fig, axes = plt.subplots(2,1,sharex=True)
        axes[0].stem(freqs, abs(spectrum))
        axes[0].set_title('Energy per band')
        axes[1].stem(freqs, numWavelets)
        axes[1].set_title('Number of reassigned analysis frequencies')
        fig.suptitle(f'Analysis for N={len(spectrum)}')

    return numWavelets


def adaptive_sswt(signal :np.ndarray, iters:int=2, method='threshold', thrsh = 1/20, otl=True, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    minimumFreq = 1 / (kwargs['ts'] * len(signal))
    maximumFreq = 1 / (kwargs['ts'] * 2)
    logger.debug("---- Mínima frecuencia para esta señal: %s", minimumFreq)
    # Transformada inicial:
    sst, cwt, freqs, scales_adp, tail = sswt(signal, **kwargs)

    for i in range(iters):
        logger.debug("******************* Iteración: %d ********************", i)
        if otl:
            spectrum = abs(cwt).sum(axis=1)/len(signal)
        else:
            spectrum = abs(sst).sum(axis=1)/len(signal)
        #del sst
        #del cwt
        numWavelets = calcNumWavelets(spectrum, freqs, method, thrsh, plotBands=False)
        
        logger.debug("---- Número de frecuncias asignadas por banda:\n%s\n----------", numWavelets)
        logger.debug("---- Frecuencias centrales de análisis:\n%s\n----------", freqs)
        limits = (freqs[:-1] + freqs[1:]) / 2
        logger.debug("---- Límites entre bandas de análisis:\n%s\n----------", limits)

        if (numWavelets==np.ones_like(freqs)).all():
            print(f'Se encontró equilibrio en {i} iteraciones')
            break 

        freqBands = np.concatenate((np.array([2*freqs[0]-limits[0]]),
                                    limits,
                                    np.array([2*freqs[-1]-limits[-1]]))) 
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
        sst, cwt, freqs, scales_adp, tail = sswt(signal, **kwargs)
        if not sst.any():
            logger.warning('ATENCIÓN SSWT contiene sólo 0s')
        
    return sst, cwt, freqs, scales_adp

    
#%%
if __name__=='__main__':
    import scipy.signal as sp
    plt.close('all')
    logging.basicConfig(filename='adaptivesswt.log', filemode='w',
                        format='%(levelname)s - %(asctime)s - %(name)s:\n %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
#%% Parámetros

    stopTime = 12
    fs = 200
    signalLen = int(stopTime * fs)    #2000

    t, ts = np.linspace(0, stopTime, signalLen, endpoint=False, retstep=True)

    compFig = plt.figure('Comparación de métodos')
    gs = compFig.add_gridspec(4,3)
    ifAx = plt.subplot(gs[1:3, 0], )
    wtAx = plt.subplot(gs[:2, 2], )
    spAx = plt.subplot(gs[:2, 1], )
    sstAx = plt.subplot(gs[2:, 1], )
    asstAx = plt.subplot(gs[2:, 2], )
    ifAx.get_shared_y_axes().join(wtAx, ifAx, spAx)
    sstAx.get_shared_y_axes().join(sstAx,asstAx)
    # compAxes = gs.subplots(sharex='col', sharey='row')
    # compAxes[1,0].remove()

    # mainFig = plt.figure('Comparación de resultados')
    # gs = mainFig.add_gridspec(2, 3)
    # mainAxes  = gs.subplots(sharex='col', sharey='row')
    # mainFig.set_tight_layout(True)
    
#%% Señal de prueba
    # longT = np.linspace(0, 20, int(signalLen*20/stopTime))
    # sig = generator.pulsePosModSig(t,pw=t[1]*3, prf=1)
    # sig = generator.hbSig(t)
    # f, sig = generator.testChirp(t, 10, 25)
    # f-=5
    # _, sig = testUpDownChirp(t,1,10)
    # f, sig = generator.quadraticChirp(t, 5, 20)
    # sig = generator.testSig(t)
    sig = generator.testSine(t,15)
    f = 15*np.ones_like(t)
    # sig = generator.delta(t, 2)
    # fqs, sig = generator.crossChrips(t, 1, 20, 4)
    # f=fqs[0]
    # sig = sig[:signalLen]
    # f = f[:signalLen]

    # mainAxes[0,0].plot(t[:len(sig)], sig)
    # mainAxes[0,0].set_title('Señal')

    # for i in range(fqs.shape[0]):
    #     compAxes[0,0].plot(t[:len(sig)], fqs[i])
    #     compAxes[1,0].plot(t[:len(sig)], fqs[i])

    ifAx.plot(t[:len(sig)], f)
    #compAxes[1,0].plot(t[:len(sig)], f)

    # compAxes[0,0].plot(t[:len(sig)], f)
    ifAx.set_title('Instaneous frequency')

    # compAxes[1,0].plot(t[:len(sig)], f)
    # compAxes[1,0].set_title('Frecuencia Instanánea')

    

#%% Transformada
    wcf = 1
    wbw = 1

    maxFreq = 30
    minFreq = 0.1
    numFreqs = 12

    config = Configuration(
        minFreq=minFreq,
        maxFreq=maxFreq,
        numFreqs=numFreqs,
        ts=ts,
        wcf=wcf,
        wbw=wbw,
        waveletBounds=(-8,8),
        threshold=sig.max()/(100),
        numProc=8,
        log=False)

    spAx.specgram(sig, NFFT=int(numFreqs), Fs=fs, scale='linear', noverlap=numFreqs-1)
    spAx.set_title('Spectrogram')
    spAx.set_ylim([minFreq,maxFreq])

    # sig=sp.hilbert(sig)
    sst, cwt, freqs, scales, tail = sswt(sig, **config.asdict())
 
    # fig = plt.figure("SSWT")
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(t, freqs)
    # ax.plot_surface(X, Y, abs(sst), cmap='viridis')

    # mainAxes[1,0].pcolormesh(t, freqs, np.abs(cwt),
    #               cmap='viridis', shading='gouraud')
    # mainAxes[1,0].set_title('Wavelet Transform')
    # mainAxes[1,1].pcolormesh(t, freqs, np.abs(sst),
    #               cmap='viridis', shading='gouraud')
    # mainAxes[1,1].set_title('Synchrosqueezing Transform')

    wtAx.pcolormesh(t, freqs, np.abs(cwt),
                  cmap='viridis', shading='gouraud')
    wtAx.set_title('Wavelet Transform')
    sstAx.pcolormesh(t, freqs, np.abs(sst),
                  cmap='viridis', shading='gouraud')
    sstAx.set_title('Synchrosqueezing Transform')


    wav = pywt.ContinuousWavelet(f'cmor{config.wbw}-{config.wcf}')
    # wav = pywt.ContinuousWavelet(f'shan{config.wbw}-{config.wcf}')
    # wav = pywt.ContinuousWavelet(f'cgau4')
    # wav = pywt.ContinuousWavelet(f'fbsp4-{config.wbw}-{config.wcf}')
    wav.lower_bound, wav.upper_bound = config.waveletBounds
    psi, x = pywt.integrate_wavelet(wav)
    print(f'\n\nLongitud de PSI: {len(psi)}\n\n')

    signalR_cwt = reconstructCWT(cwt, wav, scales, freqs)
    signalR_sst = reconstruct(sst, wav, freqs)

    deltaFreqs = np.diff(freqs, append=(2*freqs[-1] - freqs[-2]))
  
    # mainAxes[0, 1].plot(t, sig, label='Signal', alpha=0.8)
    # mainAxes[0, 1].plot(t, signalR_sst, label='Full')
    # mainAxes[0, 1].legend()
    # mainAxes[0, 1].set_title('Reconstructed signal')

    iters = 8

    mse = np.zeros(iters)
    # fig, specAx = plt.subplots(1)
    # fig.suptitle('Espectros synchrosqueezed')
    # spectrum = abs(sst).sum(axis=1)
    # specAx.plot(freqs, spectrum, label='0 Iteraciones')

    # colors = ['b','g','r','o','c','m']
    # plotFilt = False
    # for it in range(iters):
        # print('****************************************************************************************************')
        # print(f'******************************************* Iteración: {it} *******************************************')
        # if it == iters-1:
    config.plotFilt = False
    # asst, _, afreqs, scales = adaptive_sswt(sig, iters, method='proportional', thrsh=1/20, **config.asdict())
    # spectrum = abs(asst).sum(axis=1)
    # specAx.plot(freqs, spectrum, label=f'{iters} Iteraciones')
        # estFreq = freqs[spectrum.argmax()]
        # mse[it] = (estFreq - f)**2
        # logger.info('Frecuencia estimada: %s', estFreq)
    # specAx.legend()
    

    # mainAxes[1, 2].pcolormesh(t, afreqs, np.abs(asst),
    #               cmap='viridis', shading='gouraud')
    # mainAxes[1, 2].set_title(f'Adaptive SSWT - {iters} iters')

    # compAxes[1, 2].pcolormesh(t, afreqs, np.abs(asst),
    #               cmap='viridis', shading='gouraud')
    # compAxes[1, 2].set_title(f'Adaptive SSWT')

    # fig = plt.figure(f'SSWT adaptiva ({iters} iters)')
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(t, afreqs)
    # ax.plot_surface(X, Y, abs(asst), cmap='viridis')
  
    # signalR = reconstruct(asst, wav, afreqs)
    
    # mainAxes[0, 2].plot(t, signalR)
    # mainAxes[0, 2].set_title('señal reconstruída con algorimto adaptivo')

    # fig, ax = plt.subplots(1)
    # ax.plot(afreqs, np.abs(asst).sum(axis=1))
    # fig.suptitle('Espectro de la SST')

    f_sst = freqs[np.argmax(abs(sst), axis=0)]
    # f_asst = afreqs[np.argmax(abs(asst), axis=0)]

    # print(f'Tiempo de muestreo = {config.ts}')
    # fig, ax = plt.subplots(1)
    # ax.plot(np.fft.rfftfreq(len(signalR.real), d=config.ts), abs(np.fft.rfft(signalR.real)), label='Espectro Señal reconstruída')
    # ax.legend()
    # fig.suptitle('Comparación espectro')
    
    ## Minibatch
    batch_time = 2
    num_batchs = int(stopTime // batch_time)
    bLen = int(len(t[t<=(num_batchs*batch_time)]) // num_batchs)
    bPad = int(bLen * 0.8)
    sigPad = np.zeros(bPad)
    batch_iters = 3

    # batchFig = plt.figure('A-SSWT streaming')
    # bgs = batchFig.add_gridspec(1, num_batchs)
    # bgs.update(wspace=0.0)
    # batchAxes = bgs.subplots(sharey=True)
    f_batch_asst = []
    # batchFig.set_tight_layout(True)

    recoveredSignal = np.zeros_like(sig)
    

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(asstAx)
    fig1 = asstAx.get_figure()
    fig1.set_tight_layout(True)

    compositeAxes = []
    compositeAxes.append(asstAx)
    for _ in range(num_batchs - 1):
        ax2 = divider.new_horizontal(size="100%", pad=0.00)
        ax2.yaxis.set_visible(False)
        compositeAxes.append(ax2)
        fig1.add_axes(ax2)
        asstAx.get_shared_y_axes().join(asstAx, ax2)
    compositeAxes[int((num_batchs-1)//2)].set_title('Adaptive SSWT - 2s frame')
    # asstAx.get_shared_x_axes().remove(asstAx)

    for b in range(num_batchs):
        signalBatch = sig[(b*bLen)-bPad:((b+1)*bLen)+bPad] if b else sig[:((b+1)*bLen)+bPad]
        timeBatch = t[b*bLen:((b+1)*bLen)]
        # axes[b].plot(timeBatch, signalBatch)
        asst_batch, _, freqsBatch, _ = adaptive_sswt(signalBatch, batch_iters, method='proportional', thrsh=1/10, otl=True, **config.asdict())
        if b==0:
            #batchAxes[b]
            compositeAxes[b].pcolormesh(timeBatch, freqsBatch, np.abs(asst_batch[:,:bLen]),
                               cmap='viridis', shading='gouraud')
            f_batch_asst.append(freqsBatch[np.argmax(abs(asst_batch[:,:bLen]), axis=0)])
            recoveredSignal[:bLen] = reconstruct(asst_batch, wav, freqsBatch)[:bLen]
        else:
            #batchAxes[b]
            compositeAxes[b].pcolormesh(timeBatch, freqsBatch, np.abs(asst_batch[:,bPad:bLen+bPad]),
                               cmap='viridis', shading='gouraud')
            f_batch_asst.append(freqsBatch[np.argmax(abs(asst_batch[:,bPad:bLen+bPad]), axis=0)])
            recoveredSignal[b*bLen:(b+1)*bLen] = reconstruct(asst_batch, wav, freqsBatch)[bPad:bLen+bPad]


    f_batch_asst = np.array(f_batch_asst).flatten()
 
    fig, ax = plt.subplots(1)
    ax.plot(t[:len(sig)], f_sst, label='SSWT')
    ax.plot(t[:len(f_batch_asst)], f_batch_asst, label='A-SSWT (2s frame)')
    ax.plot(t[:len(sig)], f, label='Signal')
    ax.legend()
    fig.suptitle('Instantaneous frequencies')

    # fig, ax = plt.subplots(1)
    # ax.plot(t[:len(recoveredSignal)], sig[:len(recoveredSignal)], label='Singal', alpha=0.8)
    # ax.plot(t[:len(recoveredSignal)], recoveredSignal, label='Reconstructed Signal')
    # ax.legend()
    # fig.suptitle('Signal reconstruction')

    mse_sst = (f[:-30]- f_sst[:-30])**2
    # mse_asst = (f - f_asst)**2
    mse_asst_batch = (f[30:len(f_batch_asst)-30] - f_batch_asst[30:-30])**2

    mse_sst_total = mse_sst.sum() / len(mse_sst)
    # mse_asst_total = mse_asst.sum() / len(mse_asst)
    mse_asst_batch_total = mse_asst_batch.sum() / len(mse_asst_batch)

    fig, ax = plt.subplots(1)
    ax.plot(t[:len(mse_sst)], mse_sst, label=f'SST - MSE = {mse_sst_total:.3}')
    # ax.plot(t[:len(mse_asst)], mse_asst, label=f'ADPT SST - MSE = {mse_asst_total:.3}')
    ax.plot(t[30:len(f_batch_asst)-30], mse_asst_batch, label=f'ADPT SST BATCH - MSE = {mse_asst_batch_total:.3}')
    ax.legend()
    fig.suptitle('MSE(f)')


    plt.show(block=False)
    input('Presione una tecla para cerrar los gráficos...')
    plt.close('all')
