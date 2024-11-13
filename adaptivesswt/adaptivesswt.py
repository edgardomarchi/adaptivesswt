#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import queue
import threading
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as la

from .sswt import sswt
from .utils.freq_utils import getDeltaAndBorderFreqs, getScale

logger = logging.getLogger(__name__)

def _getFreqsPerBand(
    nvPerBand: np.ndarray,
    flo: float,
    fhi: float,
    freqBands: Union[np.ndarray, None] = None,
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
    frac = spectrum / quota
    frac[np.isnan(frac)] = 0
    res = frac.astype(int)

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
    assert False , f'Prop: condition should never happen. n={n}'


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

    # TODO: Remove this plot from here.
    # Since 'spectrum' and 'freqs' are arguments, and 'numWavelets' is the return value
    # this can be done outside.
    if plotBands:
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(freqs, abs(spectrum)/abs(spectrum).max())
        axes[0].set_title('Signal spectrum')
        axes[0].set_ylabel(r'$\hat{E}_{N\!t}$')
        width = np.min(np.diff(freqs)) * 0.8
        axes[1].bar(freqs, numWavelets, width=width)
        axes[1].set_ylabel(r'$K^*_k$')
        axes[0].text(
            45, 1, f'k={len(spectrum)}', ha="right", va="top")
        axes[1].set_xlabel(r'[Hz]', loc='right')

    return numWavelets


def getFreqsMPM(signal: np.ndarray, num_freqs: int, fs: float) -> np.ndarray:
    """Returns estimated signal frequencies using Matrix Pencil Method.

    Parameters
    ----------
    signal : np.ndarray
        Signal to be analyzed
    num_freqs : int
        Number of frequencies to detect
    fs : float
        Sampling frequency

    Returns
    -------
    np.ndarray
        Array containing `num_freqs` estimated frequencies
    """

    N = len(signal)
    L = int(N // 2)
    NL = N - L

    Y0 = np.empty((NL, L))
    Y1 = np.empty((NL, L))

    for i in range(L):
        Y0[:, i] = signal[(L - 1 - i) : (N - 1 - i)]
        Y1[:, i] = signal[(L - i) : (N - i)]

    U, S, VT = la.svds(Y0, num_freqs)
    Ainv = np.diag(S**-1)
    V0 = VT.T.conj()  # [:, 0:M]
    U0 = U  # [:, 0:M]
    Sol = np.dot(Ainv, np.dot(U0.T.conj(), np.dot(Y1, V0)))
    zs = np.linalg.eigvals(Sol)

    fNorm = np.angle(zs)
    fEst = np.round(fNorm[fNorm > 0] * fs / (2 * np.pi))  # type: ignore # since fNorm is an array.

    return fEst


def adaptive_sswt(
    signal: np.ndarray,
    maxIters: int = 2,
    method: str = 'proportional',
    thrsh: float = 1 / 20,
    itl: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    wab : np.ndarray
        Frequency remapping matrix
    tail : np.ndarray
        Reconstructed signal tail from padding
    """
    minimumFreq = 1 / (kwargs['ts'] * len(signal))
    logger.debug("Minimum possible frequency for this signal: %s\n", minimumFreq)

    # Initial transform:
    sst, cwt, freqs, wab, tail = sswt(signal, **kwargs)
    # Adaptation loop
    for i in range(maxIters):
        logger.debug("******************* Iteration: %d ********************\n", i)
        _, limits = getDeltaAndBorderFreqs(freqs)
        if itl:
            spectrum = ((abs(sst)) / (getScale(freqs, kwargs['ts'], kwargs['wcf'])[:, None])).sum(axis=1)  # type: ignore
            # getScale always returns ndarray in this case
        else:
            spectrum = ((abs(cwt)) / (getScale(freqs, kwargs['ts'], kwargs['wcf'])[:, None])).sum(axis=1)  # type: ignore
            # getScale always returns ndarray in this case


        numWavelets = _calcNumWavelets(
            spectrum, freqs, method, thrsh, plotBands=kwargs.get('plot_filters', False)
        )

        logger.debug("Spectrum energy normalized per band:\n%s\n", spectrum)
        logger.debug("Number of frequncies assigned per band:\n%s\n", numWavelets)
        logger.debug("Center frequencies per band:\n%s\n", freqs)
        logger.debug("Limits between bands:\n%s\n", limits)

        # When there is one frequency per band, equilibrium is found:
        if (numWavelets == np.ones_like(freqs)).all():
            logger.debug('\nEquilibrium found in %d iterations.\n', i)
            break

        freqs_adp = _getFreqsPerBand(
            numWavelets, kwargs['min_freq'], kwargs['max_freq'], freqBands=limits
        )
        scales_adp = getScale(freqs_adp, kwargs['ts'], kwargs['wcf'])

        logger.debug("Frequencies to use with CWT:\n%s\n", freqs_adp)
        kwargs['custom_scales'] = scales_adp
        sst, cwt, freqs, wab, tail = sswt(signal, **kwargs)
        if not sst.any():
            logger.warning('\nATENTION! SST contains only 0s\n\t Analized frecuencies: %s \n\n', freqs)

    return sst, freqs, wab, tail


def adaptive_sswt_overlapAndAdd(
    batchSize: int,
    signal: np.ndarray,
    maxIters: int = 2,
    method: str = 'proportional',
    thrsh: float = 1 / 20,
    itl: bool = False,
    **kwargs,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Calculates the adaptive SST for batches of the input signal using overlapAndAdd method.

    Parameters
    ----------
    batchSize : int
        Size of each batch
    signal : np.ndarray
        Signal to analyze with the ASST
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
        List of tuples containin return arrays of `adaptive_sswt()`. I.e. [(ASST matrix, frequencies, wab, tail),...].
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
            adaptive_sswt(signalBatch, maxIters, method, thrsh, itl, **kwargs)
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
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Calculates the adaptive SST for batches of the input signal using sliding window method.

    Parameters
    ----------
    batchSize : int
        Size of each batch
    signal : np.ndarray
        Signal to analyze with the ASST
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
        List of tuples containing return arrays of `adaptive_sswt()`. I.e. [(ASST matrix, frequencies, tail),...].
    """
    padding = kwargs.get('pad', 0)
    numBatchs = int(np.ceil(len(signal) / batchSize))

    jobs: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue()

    startDiscard = int(np.floor(padding / 2))
    endDiscard = int(np.ceil(padding / 2))

    threads = _create_threads(
        batchSize,
        startDiscard,
        endDiscard,
        maxIters,
        method,
        thrsh,
        itl,
        kwargs,
        jobs,
        results,
        numBatchs,
    )

    resultsList = [None] * numBatchs

    _add_jobs(signal, numBatchs, batchSize, startDiscard, endDiscard, jobs)

    try:
        jobs.join()
    except KeyboardInterrupt:  # May not work on Windows
        logger.info('... canceling adaptive batch process...')
    while not results.empty():  # Safe because all jobs have finished
        job, asst, freqs, wab, tail = results.get()  # _nowait()?
        resultsList[job] = (asst, freqs, wab, tail)

    logger.info('Batched Synchrosqueezing Done!')

    return resultsList  # type:ignore # TODO: find another way to order results


def _create_threads(
    batchSize,
    startDiscard,
    endDiscard,
    maxIters,
    method,
    thrsh,
    itl,
    config,
    jobs,
    results,
    concurrency,
):
    threads = []
    for i in range(concurrency):
        threads.append(
            threading.Thread(
                target=_worker,
                args=(
                    batchSize,
                    startDiscard,
                    endDiscard,
                    maxIters,
                    method,
                    thrsh,
                    itl,
                    config,
                    jobs,
                    results,
                ),
            )
        )
        threads[i].daemon = True
        threads[i].start()
    logger.info('Created %s threads.\n', concurrency)
    return threads


def _add_jobs(signal, numBatchs, batchSize, startDiscard, endDiscard, jobs):
    for job in range(numBatchs):
        if job == 0:
            signalBatch = signal[: batchSize + endDiscard]
        else:
            signalBatch = signal[
                (job * batchSize) - startDiscard : ((job + 1) * batchSize) + endDiscard
            ]
        jobs.put((job, signalBatch))
    logger.info('Queued %s batches.\n', numBatchs)
    return


def _worker(
    batchSize,
    startDiscard,
    endDiscard,
    maxIters,
    method,
    thrsh,
    itl,
    config,
    jobs: queue.Queue,
    results: queue.Queue,
):

    while True:
        try:
            job, signal = jobs.get()
            asst, freqs, wab, tail = adaptive_sswt(
                signal, maxIters, method, thrsh, itl, **config
            )
            if job == 0:
                results.put((job, asst[:, :batchSize], freqs, wab, tail))
            else:
                results.put(
                    (job, asst[:, startDiscard : batchSize + startDiscard], freqs, wab, tail)
                )
        finally:
            jobs.task_done()
