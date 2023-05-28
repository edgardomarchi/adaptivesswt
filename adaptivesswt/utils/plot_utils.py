#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pywt import (
    ContinuousWavelet,  # type: ignore # Pylance seems to fail finding ContinuousWavelet within pywt
)
from pywt import integrate_wavelet


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

    _, ax = plt.subplots(1, dpi=300)
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
        f = np.fft.fftfreq(max_len, ts)
        filt_fft = np.fft.fft(filt, n=max_len)
        filt_fft /= np.abs(filt_fft).max()
        ax.plot(f[f >= 0], (np.abs(filt_fft) ** 2)[f >= 0], alpha=0.6)

    sig_fft = np.abs(np.fft.fft(signal))
    sig_fft /= sig_fft.max()
    sig_f = np.fft.fftfreq(len(sig_fft), ts)
    ax.plot(sig_f[sig_f >= 0], sig_fft[sig_f >= 0], c='r')


def plot_tf_repr(tfr: np.ndarray, t: np.ndarray, f: np.ndarray, ax: Axes):
    """Helper function to plot time-frequency (TF) representations.

    Parameters
    ----------
    tfr : np.ndarray
        The TF representation matrix.
    t : np.ndarray
        Time array.
    f : np.ndarray
        Frequencies array.
    ax : Axes
        Axis to plot within.
    """
    ax.pcolormesh(t, f, np.abs(tfr), cmap='plasma', shading='gouraud')

def plot_batched_tf_repr(
    batchs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ts: float, ax: Axes):
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

    num_batchs = len(batchs)

    new_ax = ax
    for i, (asst, b_f, _, _) in enumerate(batchs):
        b_t = np.linspace(i*asst.shape[1]*ts, (i+1)*asst.shape[1]*ts, asst.shape[1], endpoint=False)
        if i != 0:
            new_ax = divider.new_horizontal(size="100%", pad=0.00)
            new_ax.yaxis.set_visible(False)
            new_ax.sharey(ax)
            fig.add_axes(new_ax)
        plot_tf_repr(asst, b_t, b_f, new_ax)
        new_ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='major',     # major ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True)
        new_ax.set_xticks([b_t[0]])
        if i == int((num_batchs - 1) // 2):
            new_ax.set_title('Adaptive SSWT - Batched')
