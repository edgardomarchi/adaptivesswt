#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script adapted from PyWavelets to print wavelets filters and longitudes.
"""
import numpy as np
import pywt
import matplotlib.pyplot as plt


plt.close('all')

fs = 2000  #Hz

wcf=0.5
wbw=2#16.0

wav = pywt.ContinuousWavelet(f'cmor{wbw}-{wcf}')
wav.lower_bound = -3
wav.upper_bound = 3

print(wav)

# print the range over which the wavelet will be evaluated
print("Continuous wavelet will be evaluated over the range [{}, {}]".format(
    wav.lower_bound, wav.upper_bound))

width = wav.upper_bound - wav.lower_bound

flo_norm = 1/fs * 0.1
fhi_norm = 1/fs * 10
maxScale = wcf / flo_norm
minScale = wcf / fhi_norm
numScales = 10
print(f'Max scale: {maxScale}, Min scale: {minScale}')
scales = np.linspace(minScale, maxScale, numScales)

precision = 7

max_len = int(np.max(scales)*width + 1)
t = np.arange(max_len)
fig, axes = plt.subplots(len(scales), 2, figsize=(12, 6))
# The following code is adapted from the internals of cwt
int_psi, x = pywt.integrate_wavelet(wav, precision=precision)
print(f'int_psi shape = {int_psi.shape} ; x shape = {x.shape}')
step = x[1] - x[0]

plt.figure('Int psi')
plt.plot(x,int_psi.real)
plt.plot(x,int_psi.imag)
plt.figure('Psi')
wavefun, y = wav.wavefun(level=precision)
plt.plot(y,wavefun.real)
plt.plot(y,wavefun.imag)
print(f'Len of int_psi = {len(x)}; len of psi = {len(y)}')

for n, scale in enumerate(scales):


    j = np.floor(
        np.arange(scale * width + 1) / (scale * step))
    print(f'J len: = {len(j)}')
    print(f'Len J* =  {len(np.arange(scale * width + 1))}')
    print(f'Arg J* =  {np.ceil(scale * width +1)}')
    if np.max(j) >= np.size(int_psi):
        j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
    j = j.astype(int)

    # normalize int_psi for easier plotting
    int_psi /= np.abs(int_psi).max()

    # discrete samples of the integrated wavelet
    filt = int_psi[j][::-1]

    # The CWT consists of convolution of filt with the signal at this scale
    # Here we plot this discrete convolution kernel at each scale.

    nt = len(filt)
    
    print(f'Scale: {scale:.3f}, longitude: {nt}, time: {nt/fs}')
    t = np.linspace(-nt//2, nt//2, nt)
    axes[n, 0].plot(t, filt.real, t, filt.imag)
    axes[n, 0].set_xlim([-max_len//2, max_len//2])
    axes[n, 0].set_ylim([-1, 1])
    axes[n, 0].text(50, 0.35, f'scale = {scale:.3f}')

    f = np.linspace(-np.pi, np.pi, max_len)
    filt_fft = np.fft.fftshift(np.fft.fft(filt, n=max_len))
    filt_fft /= np.abs(filt_fft).max()
    print(f'Escala: {scale:.3f}, |f(0)|={abs(filt_fft[np.argmin(abs(f))]):.3e}')
    axes[n, 1].plot(f, np.abs(filt_fft)**2)
    axes[n, 1].set_xlim([-np.pi, np.pi])
    axes[n, 1].set_ylim([0, 1])
    axes[n, 1].set_xticks([-np.pi, 0, np.pi])
    axes[n, 1].set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    axes[n, 1].grid(True, axis='x')
    axes[n, 1].text(np.pi/2, 0.5, 'scale = {}'.format(scale))
    axes[n, 0].set_xlabel('time (samples)')
    axes[n, 1].set_xlabel('frequency (radians)')

axes[0, 0].legend(['real', 'imaginary'], loc='upper left')
axes[0, 1].legend(['Power'], loc='upper left')
axes[0, 0].set_title('filter')
axes[0, 1].set_title(r'|FFT(filter)|$^2$')
plt.show()