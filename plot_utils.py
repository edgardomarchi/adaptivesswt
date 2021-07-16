#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from pywt import ContinuousWavelet, integrate_wavelet


def plotFilters(wav: ContinuousWavelet, scales: np.ndarray,
                 ts: float, signal: np.ndarray):
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

    fig, ax = plt.subplots(1)
    ax.grid(True, axis='x')
    ax.set_xlabel('Frequencia (Hertz)')
    fig.suptitle('Espectro de las wavelets seleccionadas')

    for scale in scales:

        # The following code is adapted from the internals of cwt
        int_psi, x = integrate_wavelet(wav, precision=8)
        step = x[1] - x[0]
        j = np.floor(
            np.arange(scale * width + 1) / (scale * step))
        if np.max(j) >= np.size(int_psi):
            j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
        j = j.astype(np.int)

        # normalize int_psi for easier plotting
        int_psi /= np.abs(int_psi).max()

        # discrete samples of the integrated wavelet
        filt = int_psi[j][:]  # [::-1]

        # The CWT consists of convolution of filt with the signal at this scale
        # Here we plot this discrete convolution kernel at each scale.
       
        #f = np.linspace(-np.pi, np.pi, max_len)
        f = np.fft.fftfreq(max_len, ts)
        filt_fft = np.fft.fft(filt, n=max_len)
        filt_fft /= np.abs(filt_fft).max()
        ax.plot(f, np.abs(filt_fft)**2, alpha=0.6)

    sigFFT = np.abs(np.fft.fft(signal))
    sigFFT /= sigFFT.max()
    sigF = np.fft.fftfreq(len(sigFFT), ts)
    ax.plot(sigF, sigFFT)