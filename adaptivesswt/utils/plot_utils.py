#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pywt import (  # type: ignore # Pylance seems to fail finding ContinuousWavelet within pywt
    ContinuousWavelet,
    integrate_wavelet,
)


def plot_cwt_filters(
    wav: ContinuousWavelet, scales: np.ndarray, ts: float, signal: np.ndarray
):
    """Plotea los espectros de las wavelets seleccionadas.

    Parameters
    ----------
    wav : pywt.ContinuousWavelet
        Template de wavelet a utilizar
    scales : np.ndarray
        Escalas para cada wavelet
    ts : float
        Período de muestreo
    signal : np.ndarray
        Señal a analizar con la CWT
    """

    ## print the range over which the wavelet will be evaluated
    width = wav.upper_bound - wav.lower_bound
    max_len = int(np.max(scales) * width + 1)

    fig, ax = plt.subplots(1, dpi=300)
    ax.grid(True, axis='x')
    ax.set_xlabel('[Hz]', loc='right')
    ax.set_title('Signal and wavelets normalized spectra')

    for scale in scales:

        # The following code is adapted from the internals of cwt
        int_psi, x = integrate_wavelet(wav, precision=8)   # type: ignore # integrate wavelet always operates with a ContinuousWavelet object
        step = x[1] - x[0]
        j = np.floor(np.arange(scale * width + 1) / (scale * step))
        if np.max(j) >= np.size(int_psi):
            j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
        j = j.astype(int)

        # normalize int_psi for easier plotting
        int_psi /= np.abs(int_psi).max()

        # discrete samples of the integrated wavelet
        filt = int_psi[j][:]  # [::-1]

        # The CWT consists of convolution of filt with the signal at this scale
        # Here we plot this discrete convolution kernel at each scale.

        # f = np.linspace(-np.pi, np.pi, max_len)
        f = np.fft.fftfreq(max_len, ts)
        filt_fft = np.fft.fft(filt, n=max_len)
        filt_fft /= np.abs(filt_fft).max()
        ax.plot(f[f >= 0], (np.abs(filt_fft) ** 2)[f >= 0], alpha=0.6)

    sigFFT = np.abs(np.fft.fft(signal))
    sigFFT /= sigFFT.max()
    sigF = np.fft.fftfreq(len(sigFFT), ts)
    ax.plot(sigF[sigF >= 0], sigFFT[sigF >= 0], c='r')


def plotSSWTminiBatchs(
    batchs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ax: Axes):
    """Plots in a packed set of axes the results of the batched ASSWT

    Parameters
    ----------
    batchs : List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of tuples containing (ASSWT, Frequencies, tail)
    ax : Axis
        Matplotlib Axis where function will plot
    """

    divider = make_axes_locatable(ax)
    fig = ax.get_figure()

    numBatchs = len(batchs)

    newAx = ax
    for i, (asswt, freqs, _, _) in enumerate(batchs):
        batchTime = np.arange(asswt.shape[1])
        if i != 0:
            newAx = divider.new_horizontal(size="100%", pad=0.00)
            newAx.yaxis.set_visible(False)
            newAx.sharey(ax)
            fig.add_axes(newAx)
            #ax.get_shared_y_axes().join(ax, newAx)
        newAx.pcolormesh(
            batchTime, freqs, np.abs(asswt), cmap='plasma', shading='gouraud')
        newAx.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        if i == int((numBatchs - 1) // 2):
            newAx.set_title('Adaptive SSWT - Batched')
