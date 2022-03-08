#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

import pywt
from typing import Tuple

from .configuration import Configuration

from .utils import signal_utils as generator
from .utils.freq_utils import getDeltaAndBorderFreqs, calcScalesAndFreqs, getScale
from .sswt import sswt, reconstruct, reconstructCWT


def _getFreqsPerBand(nvPerBand:np.ndarray, flo:float, fhi:float,
                     freqBands:np.ndarray=None) -> np.ndarray:
    """Returns frequencies per band according to the number of voices for each one.

    Parameters
    ----------
    nvPerBand : np.ndarray
        Number of voices per band
    flo : float
        Minimum frequency
    fhi : float
        Maximum frequency
    freqBands : np.ndarray, optional
        Limit frequencies between bands, by default None

    Returns
    -------
    np.ndarray
        Frequencies to use with CWT ordered from flo to fhi
    """
    if freqBands is None:
        freqs = np.linspace(flo, fhi, len(nvPerBand)+1, endpoint=True)
    else:
        assert len(freqBands) == len(nvPerBand)+1,\
            "ERROR: Number of frequency bands is not coherent with the number of voices per band!"
        freqs = freqBands
    newFreqs = np.zeros(np.array(nvPerBand).sum())
    nv = 0
    numBands = len(nvPerBand)
    for i in range(numBands):
        bFreqs = np.linspace(freqs[i], freqs[i+1], nvPerBand[i]+2, endpoint=True)[1:-1]
        newFreqs[nv:nv+nvPerBand[i]] = bFreqs
        nv += nvPerBand[i]
    return newFreqs


def _proportional(nv: int, spectrum: np.ndarray) -> np.ndarray:
    """Assign nv frequencies proportionally to the spectrum using Hagenbach-Bischoff quota

    Parameters
    ----------
    nv : int
        Number of frequencies to assign
    spectrum : np.ndarray 
        Array of float weighting each band
    
    Returns
    -------
    np.ndarray
        Array of ints seats allocated to each band
    """
    quota = sum(spectrum) / (1 + nv)
    frac = np.array([freq/quota for freq in spectrum])
    res = np.array([int(f) for f in frac])
    n = nv - res.sum() # number of frequencies remaining to allocate
    if n==0: return res #done
    if n<0: return [min(x,nv) for x in res]
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


def _calcNumWavelets(spectrum:np.ndarray, freqs:np.ndarray, method='proportional', thrsh=1/20,
                    plotBands:bool=True) -> np.ndarray:
    """Calculates the number of frecuencies to allocate within each band.

    Parameters
    ----------
    spectrum : np.ndarray
        Spectrum magnitude of the signal
    freqs : np.ndarray
        Frequencies (only for plotting)
    method : {'proportional', 'threshold'}, optional
        Frequency reallocation method
    plotBands : bool, optional
        If True, plots magnitude of wavelet filter's frequency response

    Returns
    -------
    np.ndarray
        Number of assigned frequencies per band
    """
    if method=='proportional':
        numWavelets = _proportional(len(spectrum), spectrum)
    elif method=='threshold':
        numWavelets = np.where(spectrum > (spectrum.max() * thrsh), 1, 0)
        numUnusedWavelets = len(spectrum) - numWavelets.sum()  
        numWavelets += _proportional(numUnusedWavelets, spectrum)
    else:  # For now, proportional is the default
        numWavelets = _proportional(len(spectrum), spectrum)

    rem = len(spectrum) - numWavelets.sum()
    logger.debug('Center frequencies: \n %s', freqs)
    logger.debug('Remainder: %d', rem)

    if plotBands:
        fig, axes = plt.subplots(2,1,sharex=True)
        axes[0].stem(freqs, abs(spectrum))
        axes[0].set_title('Energy per band')
        axes[1].stem(freqs, numWavelets)
        axes[1].set_title('Number of reassigned analysis frequencies')
        fig.suptitle(f'Analysis for N={len(spectrum)}')

    return numWavelets


def adaptive_sswt(signal :np.ndarray, iters :int=2, method :str='proportional',
                  thrsh :float=1/20, otl :bool=True, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs an adaptive sswt according energy distribution across signal spectrum.

    Parameters
    ----------
    signal: np.ndarray
        Signal to analyze
    iters : int, optional
        Maximum number of adaptation iterations, by default 2
    method: {'proportional','threshold'}, optional
        Frequency reallocation method. Either 'threshold' or 'proportional', by default 'proportional'.
    thrsh: float, optional
        Detection threshold for 'threshold' method, by default '1/20'.
    otl: bool, optional
        True if off-the-loop synchrosqueezing is performed, by default True.
    **kwargs: dict
        Same arguments than `sswt()`

    Returns
    -------
    St : np.ndarray
        Matrix with the Adaptive Synchrosqueezing Transform
    freqs : np.ndarray
        Array with analysis frequencies
    tail : np.ndarray
        Reconstructed signal tail from padding
    """
    minimumFreq = 1 / (kwargs['ts'] * len(signal))
    logger.debug("Minimum possible frequency for this signal: %s\n", minimumFreq)

    # Initial transform:
    sst, freqs, tail = sswt(signal, **kwargs)
    # Adaptation loop
    for i in range(iters):
        logger.debug("******************* Iteration: %d ********************\n", i)
        _, limits = getDeltaAndBorderFreqs(freqs)
        if otl:
            spectrum = abs(cwt).sum(axis=1)
        else:
            spectrum = abs(sst).sum(axis=1)
        
        spectrum /= spectrum.max()
 
        numWavelets = _calcNumWavelets(spectrum, freqs, method, thrsh, plotBands=False)
        
        logger.debug("Number of frequncies assigned per band:\n%s\n", numWavelets)
        logger.debug("Center frequencies per band:\n%s\n", freqs)
        logger.debug("Limits between bands:\n%s\n", limits)

        # When there is one frequency per band, equilibrium is found:
        if (numWavelets==np.ones_like(freqs)).all():
            logger.debug('\nEquilibrium found in %d iterations.\n',i)
            break 

        freqs_adp = _getFreqsPerBand(numWavelets, kwargs['minFreq'], kwargs['maxFreq'],
                                     freqBands=limits)
        scales_adp = getScale(freqs_adp, ts, wcf)

        logger.debug("Frequencies to use with CWT:\n%s\n", freqs_adp)
        kwargs['custom_scales'] = scales_adp
        sst, freqs, tail = sswt(signal, **kwargs)
        if not sst.any():
            logger.warning('ATENTION! SSWT contains only 0s')
        
    return sst, freqs, tail

def adaptive_sswt_miniBatch(batchSize: int, signal :np.ndarray, iters :int=2, method :str='proportional',
                thrsh :float=1/20, otl :bool=True, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tail = np.zeros(batchSize)
    rTail = np.zeros(1)

    num_batchs = np.ceil(len(signal)/batchSize, dtype=int)

    for b in range(num_batchs):
        # Checkear el último chunk
#        signalBatch = sig[(b*batchSize)-bPad:((b+1)*batchSize)+bPad] if b else sig[:((b+1)*batchSize)+bPad]
        if len(rTail) < len(tail):
            tail[:len(rTail)] = rTail
        else:
            tail = rTail[:len(tail)]

        signalBatch = sig[b*batchSize:((b+1)*batchSize)] + tail[:batchSize]

        timeBatch = t[b*batchSize:((b+1)*batchSize)]
        # axes[b].plot(timeBatch, signalBatch)
        asst_batch, freqsBatch, rTail = adaptive_sswt(signalBatch, iters, method=method, thrsh=thrsh, otl=otl, **config.asdict())
        if b==0:
            #batchAxes[b]
            compositeAxes[b].pcolormesh(timeBatch, freqsBatch, np.abs(asst_batch), #[:,:batchSize]),
                            cmap='viridis', shading='gouraud')
            f_batch_asst.append(freqsBatch[np.argmax(abs(asst_batch), #[:,:batchSize]), 
                                                    axis=0)])
            recoveredSignal[:batchSize] = reconstruct(asst_batch, config.C_psi, freqsBatch) #[:batchSize]
        else:
            #batchAxes[b]
            compositeAxes[b].pcolormesh(timeBatch, freqsBatch, np.abs(asst_batch), #[:,bPad:batchSize+bPad]),
                            cmap='viridis', shading='gouraud')
            f_batch_asst.append(freqsBatch[np.argmax(abs(asst_batch),#[:,bPad:batchSize+bPad]),
                                                    axis=0)])
            recoveredSignal[b*batchSize:(b+1)*batchSize] = reconstruct(asst_batch, config.C_psi, freqsBatch) #[bPad:batchSize+bPad]


    f_batch_asst = np.array(f_batch_asst).flatten()
    return
    
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
    f, sig = generator.quadraticChirp(t, 5, 25)
    # sig = generator.testSig(t)
    # sig = generator.testSine(t,15)
    # f = 15*np.ones_like(t)
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
        waveletBounds=(-3,3),
        threshold=sig.max()/(100),
        numProc=4)

    spAx.specgram(sig, NFFT=int(numFreqs), Fs=fs, scale='linear', noverlap=numFreqs-1)
    spAx.set_title('Spectrogram')
    spAx.set_ylim([minFreq,maxFreq])

    # sig=sp.hilbert(sig)
    scales, _, _, _ = calcScalesAndFreqs(ts, config.wcf, config.minFreq, config.maxFreq, config.numFreqs)
    cwt, freqsCWT = pywt.cwt(sig, scales, config.wav, sampling_period=ts, method='fft')
    sst, freqs, tail = sswt(sig, **config.asdict())
 
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
    signalR_sst = reconstruct(sst, config.C_psi, freqs)

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
    batch_iters = 5

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

    tail = np.zeros(bLen)
    rTail = np.zeros(1)
    config.pad=True

    for b in range(num_batchs):
#        signalBatch = sig[(b*bLen)-bPad:((b+1)*bLen)+bPad] if b else sig[:((b+1)*bLen)+bPad]
        if len(rTail) < len(tail):
            tail[:len(rTail)] = rTail
        else:
            tail = rTail[:len(tail)]

        signalBatch = sig[b*bLen:((b+1)*bLen)] + tail[:bLen]

        timeBatch = t[b*bLen:((b+1)*bLen)]
        # axes[b].plot(timeBatch, signalBatch)
        asst_batch, freqsBatch, rTail = adaptive_sswt(signalBatch, batch_iters, method='proportional', thrsh=1/10, otl=False, **config.asdict())
        if b==0:
            #batchAxes[b]
            compositeAxes[b].pcolormesh(timeBatch, freqsBatch, np.abs(asst_batch), #[:,:bLen]),
                               cmap='viridis', shading='gouraud')
            f_batch_asst.append(freqsBatch[np.argmax(abs(asst_batch), #[:,:bLen]), 
                                                     axis=0)])
            recoveredSignal[:bLen] = reconstruct(asst_batch, config.C_psi, freqsBatch) #[:bLen]
        else:
            #batchAxes[b]
            compositeAxes[b].pcolormesh(timeBatch, freqsBatch, np.abs(asst_batch), #[:,bPad:bLen+bPad]),
                               cmap='viridis', shading='gouraud')
            f_batch_asst.append(freqsBatch[np.argmax(abs(asst_batch),#[:,bPad:bLen+bPad]),
                                                     axis=0)])
            recoveredSignal[b*bLen:(b+1)*bLen] = reconstruct(asst_batch, config.C_psi, freqsBatch) #[bPad:bLen+bPad]


    f_batch_asst = np.array(f_batch_asst).flatten()
 
    #_ = adaptive_sswt_miniBatch(batchSize=2*fs)

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
