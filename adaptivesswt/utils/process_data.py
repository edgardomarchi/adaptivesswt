#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:26:25 2021

@author: edgardo
"""
import logging
import time as clock
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp

from adaptivesswt.adaptivesswt import adaptive_sswt, adaptive_sswt_slidingWindow
from adaptivesswt.configuration import Configuration
from adaptivesswt.sswt import sswt
from adaptivesswt.utils.import_utils import MeasurementData
from adaptivesswt.utils.plot_utils import plot_batched_tf_repr, plot_tf_repr

logger = logging.getLogger(__name__)

def extractPhase(data: MeasurementData) -> Tuple[np.ndarray, np.ndarray]:
    """Exctracts the unwrapped phase of the radar data.

    Parameters
    ----------
    data : MeasurementData
        Measurement data structure

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Signal phase and time axis (respectively)
    """
    signal = data.radarI + 1j*data.radarQ
    signalPhase = np.unwrap(np.angle(signal))

    stop = len(signal) / data.fs
    time = np.linspace(0, stop, len(signal))

    return signalPhase, time

def intDecimate(signal: np.ndarray, fs: float,
                fpcg: float, fpulse: float, fresp:float) -> Tuple[Tuple[np.ndarray, float],
                                                                  Tuple[np.ndarray, float],
                                                                  Tuple[np.ndarray, float]]:
    """Returns the integer-rate sub-sampled signals for pcg, pulse and respiration

    Parameters
    ----------
    signal : np.ndarray
        Data signal to be decimated
    fs : float
        Data signal sampling frequency
    fpcg : float
        Desired PCG signal sampling frequency
    fpulse : float
        Desired Pulse signal sampling frequency
    fresp : float
        Desired Respiration signal sampling frequency

    Returns
    -------
    Tuple[Tuple[np.ndarray, float], Tuple[np.ndarray, float], Tuple[np.ndarray, float]]
        Tuples of PCG, pulse and respiration pairs of (signal, sampling frequency) respectively
    """
    # PCG
    pcgDecRate = int(fs/fpcg)
    pcgFs = fs / pcgDecRate
    pcgDecSignal = sp.decimate(signal, pcgDecRate,ftype='fir')
    # Pulse
    pulseDecRate = int(pcgFs/fpulse)
    pulseFs = pcgFs / pulseDecRate
    pulseDecSignal = sp.decimate(pcgDecSignal, pulseDecRate,ftype='fir')
    # Respiration
    respDecRate = int(pulseFs/fresp)
    respFs = pulseFs / respDecRate
    respDecSignal = sp.decimate(pulseDecSignal, respDecRate,ftype='fir')
    logger.debug('Fs: %s, DRPCG: %s, fsPcg: %s, DRPulse: %s, fsPulse: %s, DRResp: %s, fsResp: %s',
                 fs, pcgDecRate, pcgFs, pulseDecRate, pulseFs, respDecRate, respFs)

    return (pcgDecSignal, pcgFs), (pulseDecSignal, pulseFs), (respDecSignal, respFs)


def analyze(signal: np.ndarray, config: Configuration,
            iters: int=0, method: str='threshold', threshold: float = 1/100, itl: bool=False,
            bLen: int=256, plot: bool=True, tsst=False
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, Union[plt.Figure, None]]:
    """Analyzes the signal with the adaptive SST

    Parameters
    ----------
    signal : np.ndarray
        Signal to analyze
    config : configuration
        Configuration parameters of the transform
    iters : int, optional
        Number of iterations performed by the algorithm, by default 0
    method : {'threshold', 'proportional'}, optional
        ASST frequency reallocation method, by default 'threshold'
    threshold : float, optional
        Threshold for 'threshold' method, by default 1/100
    itl: bool, optional
        In-the-loop synchrosqueezing if True, else Off-the-loop, by default False
    bLen: int, optional
        The number of samples of each batch, by default 256
    plot : bool, optional
        'True' to plot CWT and SST, by default False

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, list, Union[plt.Figure, None]]
        Tuple containing the SST, the ASST, the analysis frequencies, the batchs of BASST, and if `plot = True` the figure with TF representations
    """
    time = np.linspace(0, len(signal)*config.ts, len(signal))
    sst, _, freqs, _, _ = sswt(signal, **config.asdict(), tsst=tsst)
    pre = clock.time()
    asst, afreqs, _, _ = adaptive_sswt(signal, iters, method, threshold, itl, **config.asdict(), tsst=tsst)
    print(f'Real time taken to comute ASST = {clock.time()-pre}')
    batchs = adaptive_sswt_slidingWindow(
        bLen, signal, iters, method, threshold, itl, **config.asdict(), tsst=tsst
    )

    print(f'Blen = {bLen}, Batchs = {len(batchs)}')
    fig = None
    if plot:
        fig = plt.figure(figsize=(17/2.54,6/2.54), dpi=300)
        gs = fig.add_gridspec(1, 3)
        stAx = plt.subplot(gs[0, 0],)
        asAx = plt.subplot(gs[0, 1],)
        baAx = plt.subplot(gs[0,2],)
        stAx.get_shared_y_axes().join(stAx, asAx, baAx)
        plot_tf_repr(sst,time, freqs, stAx)
        stAx.set_title('SST')
        plot_tf_repr(asst, time, afreqs, asAx)
        asAx.set_title('ASST')
        plot_batched_tf_repr(batchs, config.ts, baAx)

        gs.tight_layout(fig)#, rect=[0, 0, 0.8, 1])

    return sst, asst, afreqs, batchs, fig
