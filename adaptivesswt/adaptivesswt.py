#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy.signal as sp

from .configuration import Configuration
from .sswt import reconstruct, reconstructCWT, sswt
from .utils import signal_utils as generator
from .utils.freq_utils import calcScalesAndFreqs, getDeltaAndBorderFreqs, getScale
from .utils.plot_utils import plotSSWTminiBatchs


def _getFreqsPerBand(
    nvPerBand: np.ndarray, flo: float, fhi: float, freqBands: np.ndarray = None
) -> np.ndarray:
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
        freqs = np.linspace(flo, fhi, len(nvPerBand) + 1, endpoint=True)
    else:
        assert (
            len(freqBands) == len(nvPerBand) + 1
        ), "ERROR: Number of frequency bands is not coherent with the number of voices per band!"
        freqs = freqBands
    newFreqs = np.zeros(np.array(nvPerBand).sum())
    nv = 0
    numBands = len(nvPerBand)
    for i in range(numBands):
        bFreqs = np.linspace(freqs[i], freqs[i + 1], nvPerBand[i] + 2, endpoint=True)[
            1:-1
        ]
        newFreqs[nv : nv + nvPerBand[i]] = bFreqs
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
    frac = np.array([freq / quota for freq in spectrum])
    res = np.array([int(f) for f in frac])
    n = nv - res.sum()  # number of frequencies remaining to allocate
    if n == 0:
        return res  # done
    if n < 0:
        return np.array([min(x, nv) for x in res])
    # give the remaining wavelets to the n frequencies with the largest remainder
    remainders = [ai - bi for ai, bi in zip(frac, res)]
    limit = sorted(remainders, reverse=True)[n - 1]
    # n frequencies with remainter larger than limit get an extra wavelet
    for i, r in enumerate(remainders):
        if r >= limit:
            res[i] += 1
            n -= 1  # attempt to handle perfect equality
            if n == 0:
                return res  # done
    assert False  # should never happen


def _calcNumWavelets(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    method='proportional',
    thrsh=1 / 20,
    plotBands: bool = True,
) -> np.ndarray:
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
    if method == 'proportional':
        numWavelets = _proportional(len(spectrum), spectrum)
    elif method == 'threshold':
        numWavelets = np.where(spectrum > (spectrum.max() * thrsh), 1, 0)
        numUnusedWavelets = len(spectrum) - numWavelets.sum()
        numWavelets += _proportional(numUnusedWavelets, spectrum)
    else:  # For now, proportional is the default
        numWavelets = _proportional(len(spectrum), spectrum)

    rem = len(spectrum) - numWavelets.sum()
    logger.debug('Center frequencies: \n %s', freqs)
    logger.debug('Remainder: %d', rem)

    if plotBands:
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].stem(freqs, abs(spectrum))
        axes[0].set_title('Energy per band')
        axes[1].stem(freqs, numWavelets)
        axes[1].set_title('Number of reassigned analysis frequencies')
        fig.suptitle(f'Analysis for N={len(spectrum)}')

    return numWavelets


def adaptive_sswt(
    signal: np.ndarray,
    maxIters: int = 2,
    method: str = 'proportional',
    thrsh: float = 1 / 20,
    itl: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs an adaptive sswt according energy distribution across signal spectrum.

    Parameters
    ----------
    signal: np.ndarray
        Signal to analyze
    maxIters : int, optional
        Maximum number of iterations of the adaptive algorithm, by default 2
    method: {'proportional','threshold'}, optional
        Frequency reallocation method. Either 'threshold' or 'proportional', by default 'proportional'.
    thrsh: float, optional
        Detection level for 'threshold' method, by default '1/20'.
    itl: bool, optional
        True if in-the-loop synchrosqueezing is performed, by default False.
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
    sst, cwt, freqs, tail = sswt(signal, **kwargs)
    # Adaptation loop
    for i in range(maxIters):
        logger.debug("******************* Iteration: %d ********************\n", i)
        _, limits = getDeltaAndBorderFreqs(freqs)
        if itl:
            spectrum = ((abs(sst) ** 2) / (getScale(freqs, kwargs['ts'], kwargs['wcf'])[:, None])).sum(axis=1)  # type: ignore
        else:  # getScale always returns ndarray in this case
            spectrum = ((abs(cwt) ** 2) / (getScale(freqs, kwargs['ts'], kwargs['wcf'])[:, None])).sum(axis=1)  # type: ignore
            # getScale always returns ndarray in this case

        spectrum /= spectrum.max()

        numWavelets = _calcNumWavelets(spectrum, freqs, method, thrsh, plotBands=False)

        logger.debug("Spectrum energy normalized per band:\n%s\n", spectrum)
        logger.debug("Number of frequncies assigned per band:\n%s\n", numWavelets)
        logger.debug("Center frequencies per band:\n%s\n", freqs)
        logger.debug("Limits between bands:\n%s\n", limits)

        # When there is one frequency per band, equilibrium is found:
        if (numWavelets == np.ones_like(freqs)).all():
            logger.debug('\nEquilibrium found in %d iterations.\n', i)
            break

        freqs_adp = _getFreqsPerBand(
            numWavelets, kwargs['minFreq'], kwargs['maxFreq'], freqBands=limits
        )
        scales_adp = getScale(freqs_adp, kwargs['ts'], kwargs['wcf'])

        logger.debug("Frequencies to use with CWT:\n%s\n", freqs_adp)
        kwargs['custom_scales'] = scales_adp
        sst, cwt, freqs, tail = sswt(signal, **kwargs)
        if not sst.any():
            logger.warning('ATENTION! SSWT contains only 0s')

    return sst, freqs, tail


def adaptive_sswt_overlapAndAdd(
    batchSize: int,
    signal: np.ndarray,
    maxIters: int = 2,
    method: str = 'proportional',
    thrsh: float = 1 / 20,
    itl: bool = False,
    **kwargs,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculates the adaptive SSWT for batches of the input signal using overlapAndAdd method.

    Parameters
    ----------
    batchSize : int
        Size of each batch
    signal : np.ndarray
        Signal to analyze with the A-SSWT
    maxIters : int, optional
        Maximum number of iterations of the adaptive algorithm, by default 2
    method : {'proportional','threshold'}, optional
        Frequency reallocation method. Either 'threshold' or 'proportional', by default 'proportional'.
    thrsh : float, optional
        Detection level for 'threshold' method, by default 1/20
    itl : bool, optional
        True if in-the-loop synchrosqueezing is performed, by default False.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of tuples containin return arrays of `adaptive_sswt()`. I.e. [(ASSWT matrix, frequencies, tail),...].
    """

    tail = np.zeros(batchSize)
    rTail = np.zeros(1)

    num_batchs = int(np.ceil(len(signal) / batchSize))

    results = []

    for b in range(num_batchs):

        if len(rTail) < len(tail):
            tail[: len(rTail)] = rTail
        else:
            tail = rTail[: len(tail)]

        signalBatch = signal[b * batchSize : (b + 1) * batchSize] + tail[:batchSize]

        results.append(
            adaptive_sswt(
                signalBatch,
                maxIters,
                method=method,
                thrsh=thrsh,
                itl=itl,
                **config.asdict(),
            )
        )

    return results

def adaptive_sswt_slidingWindow(
    batchSize: int,
    signal: np.ndarray,
    maxIters: int = 2,
    method: str = 'proportional',
    thrsh: float = 1 / 20,
    itl: bool = False,
    **kwargs,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculates the adaptive SSWT for batches of the input signal using sliding window method.

    Parameters
    ----------
    batchSize : int
        Size of each batch
    signal : np.ndarray
        Signal to analyze with the A-SSWT
    maxIters : int, optional
        Maximum number of iterations of the adaptive algorithm, by default 2
    method : {'proportional','threshold'}, optional
        Frequency reallocation method. Either 'threshold' or 'proportional', by default 'proportional'.
    thrsh : float, optional
        Detection level for 'threshold' method, by default 1/20
    itl : bool, optional
        True if in-the-loop synchrosqueezing is performed, by default False.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of tuples containin return arrays of `adaptive_sswt()`. I.e. [(ASSWT matrix, frequencies, tail),...].
    """
    padding = kwargs.get('pad',0)
    num_batchs = int(np.ceil((len(signal) - padding) / batchSize))

    results = []
    startDiscard = int(np.floor(padding/2))
    endDiscard = int(np.ceil(padding/2))

    for b in range(num_batchs):

        if b == 0:  # First batch has no startDiscard
            signalBatch = signal[:batchSize + endDiscard]
            asst, freqs, tail = adaptive_sswt(signalBatch, maxIters, method, thrsh, itl, **config.asdict())
            results.append((asst[:, : batchSize], freqs, tail))
        else:
            signalBatch = signal[(b * batchSize) - startDiscard : ((b + 1) * batchSize) + endDiscard]
            asst, freqs, tail = adaptive_sswt(signalBatch, maxIters, method, thrsh, itl, **config.asdict())
            results.append((asst[:, startDiscard: startDiscard + batchSize], freqs, tail))

    return results


#%%
if __name__ == '__main__':

    import scipy.signal as sp

    plt.close('all')
    logging.basicConfig(
        filename='adaptivesswt.log',
        filemode='w',
        format='%(levelname)s - %(asctime)s - %(name)s:\n %(message)s',
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    #%% Parameters

    stopTime = 12
    fs = 200
    signalLen = int(stopTime * fs)

    t, ts = np.linspace(0, stopTime, signalLen, endpoint=False, retstep=True)

    #%% Setup Figures

    compFig = plt.figure('Method comparison')
    gs = compFig.add_gridspec(2, 3)
    ifAx = plt.subplot(
        gs[0, 0],
    )
    wtAx = plt.subplot(
        gs[0, 2],
    )
    spAx = plt.subplot(
        gs[0, 1],
    )
    sstAx = plt.subplot(
        gs[1, 0],
    )
    asstAx = plt.subplot(
        gs[1, 1],
    )
    bAsstAx = plt.subplot(
        gs[1, 2],
    )
    ifAx.get_shared_y_axes().join(wtAx, ifAx, spAx)
    sstAx.get_shared_y_axes().join(sstAx, asstAx, bAsstAx)
    gs.tight_layout(compFig)

    #%% Test signals
    # longT = np.linspace(0, 20, int(signalLen*20/stopTime))
    # sig = generator.pulsePosModSig(t,pw=t[1]*3, prf=1)
    # sig = generator.hbSig(t)
    # f, sig = generator.testChirp(t, 10, 25)
    # f-=5
    # _, sig = testUpDownChirp(t,1,10)
    f, sig = generator.quadraticChirp(t, 40, 30)
    # sig = generator.testSig(t)
    # sig = generator.testSine(t,15)
    # f = 15*np.ones_like(t)
    # sig = generator.delta(t, 2)
    # fqs, sig = generator.crossChrips(t, 1, 20, 4)
    # f=fqs[0]
    # sig = sig[:signalLen]
    # f = f[:signalLen]

    ifAx.plot(t[: len(sig)], f)
    ifAx.set_title('Instaneous frequency')

    #%% SSWT, CWT and Spectrogram
    wcf = 1
    wbw = 0.5

    maxFreq = 50
    minFreq = 20
    numFreqs = 12

    config = Configuration(
        minFreq=minFreq,
        maxFreq=maxFreq,
        numFreqs=numFreqs,
        ts=ts,
        wcf=wcf,
        wbw=wbw,
        waveletBounds=(-3, 3),
        threshold=sig.max() / (100),
        numProc=4,
    )

    spAx.specgram(sig, NFFT=int(numFreqs), Fs=fs, scale='linear', noverlap=numFreqs - 1)
    spAx.set_title('Spectrogram')
    spAx.set_ylim([minFreq, maxFreq])

    sst, cwt, freqs, tail = sswt(sig, **config.asdict())

    wtAx.pcolormesh(t, freqs, np.abs(cwt), cmap='viridis', shading='gouraud')
    wtAx.set_title('Wavelet Transform')

    sstAx.pcolormesh(t, freqs, np.abs(sst), cmap='viridis', shading='gouraud')
    sstAx.set_title('Synchrosqueezing Transform')

    scales = getScale(freqs, config.ts, config.wcf)
    signalR_cwt = reconstructCWT(cwt, config.wav, scales, freqs)  # type: ignore # scales is always a ndarray
    signalR_sst = reconstruct(sst, config.C_psi, freqs)

    f_sst = freqs[np.argmax(abs(sst), axis=0)]
    f_cwt = freqs[np.argmax(abs(cwt), axis=0)]

    #%% Adaptive SSWT and minibatch A SSWT
    maxIters = 5
    threshold = config.threshold * 2
    method = 'proportional'
    itl = True

    asst, aFreqs, tail = adaptive_sswt(
        sig, maxIters, method, threshold, itl, **config.asdict()
    )

    f_asst = aFreqs[np.argmax(abs(asst), axis=0)]
    asstAx.pcolormesh(t, aFreqs, np.abs(asst), cmap='viridis', shading='gouraud')
    asstAx.set_title('Adaptive SSWT')

    signalR_asst = reconstruct(asst, config.C_psi, aFreqs)

    ## Minibatch
    batch_time = 2
    num_batchs = int(stopTime // batch_time)
    bLen = int(len(t[t <= (num_batchs * batch_time)]) // num_batchs)
    bPad = int(bLen * 0.9)
    config.pad = bPad
    bMaxIters = 5

    batchs = adaptive_sswt_slidingWindow(
        bLen, sig, bMaxIters, method, threshold, itl, **config.asdict()
    )
    plotSSWTminiBatchs(batchs, bAsstAx)

    fBatchList = []
    # Recover instantaneous frequencies:
    for (asswt, freqs, tail) in batchs:
        fBatchList.append(freqs[np.argmax(abs(asswt), axis=0)])

    f_batch = np.array(fBatchList[1:-1]).flatten()
    f_batch = np.concatenate((fBatchList[0], f_batch, fBatchList[-1]))

    #%% Instantaneous frequencies across methods
    fig, ax = plt.subplots(1)
    ax.plot(t[: len(sig)], f_cwt, label='CWT')
    ax.plot(t[: len(sig)], f_sst, label='SSWT')
    ax.plot(t[: len(sig)], f_asst, label='A-SSWT')
    ax.plot(t[: len(f_batch)], f_batch, label=f'A-SSWT ({batch_time}s batchs)')
    ax.plot(t[: len(sig)], f, label='Signal')
    ax.legend()
    fig.suptitle('Instantaneous frequencies')

    fig, ax = plt.subplots(1)
    ax.plot(
        t[: len(signalR_sst)],
        sig[: len(signalR_sst)],
        label='Original Signal',
        alpha=0.8
    )
    ax.plot(
        t[: len(signalR_sst)],
        signalR_sst,
        label='SSWT Signal'
    )
    ax.plot(
        t[: len(signalR_asst)],
        signalR_asst,
        label='A-SSWT Signal'
    )
    ax.legend()
    fig.suptitle('Signal reconstruction')

    mse_cwt = (f - f_cwt) ** 2
    mse_sst = (f - f_sst) ** 2
    mse_asst = (f - f_asst) ** 2
    mse_asst_batch = (f[: len(f_batch)] - f_batch) ** 2

    mse_cwt_total = mse_cwt.sum() / len(mse_cwt)
    mse_sst_total = mse_sst.sum() / len(mse_sst)
    mse_asst_total = mse_asst.sum() / len(mse_asst)
    mse_asst_batch_total = mse_asst_batch.sum() / len(mse_asst_batch)

    fig, ax = plt.subplots(1)
    ax.plot(t[: len(mse_cwt)], mse_cwt, label=f'CWT - MSE = {mse_cwt_total:.3}')
    ax.plot(t[: len(mse_sst)], mse_sst, label=f'SST - MSE = {mse_sst_total:.3}')
    ax.plot(t[: len(mse_asst)], mse_asst, label=f'ADPT SST - MSE = {mse_asst_total:.3}')
    ax.plot(
        t[: len(f_batch)],
        mse_asst_batch,
        label=f'A-SST Batch - MSE = {mse_asst_batch_total:.3}',
    )
    ax.legend()
    fig.suptitle('MSE(f)')

    plt.show(block=False)
    input('Press enter to close plots...')
    plt.close('all')
