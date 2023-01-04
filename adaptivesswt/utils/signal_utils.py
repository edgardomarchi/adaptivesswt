#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
from scipy.signal import chirp as spchirp


def testSig(t :np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Test signal as Fig.1 of "The Synchrosqueezing Transform: A EMD like tool" - Deaubechies.

    Parameters
    ----------
    t : np.ndarray
        Time array

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of instantaneous frequencies and signal samples respectively
    """
    ventana1 = t < (5/2)*np.pi
    f1 = 2
    s1 = (0.*t + np.cos(2*np.pi*f1*t)) * ventana1
    if1 = f1* ventana1
    ventana2 = t > 2*np.pi
    s2 = np.cos((2/3)*((t-10)**3 - (2*np.pi - 10)**3)
                + 10*(t - 2*np.pi)) * ventana2
    if2 = ((t**3)/(3*np.pi)-(150/np.pi)*(t**2)+(20/np.pi)*t)*ventana2
    return if1 + if2, s1 + s2

def hbSig(t :np.ndarray, A: float=1,
          ab: float=0.018, fb: float=0.2,
          ah: float=0.004, fh: float=1) -> np.ndarray:
    """Two IMF components signal, similar to breathing + heart pulse

    Parameters
    ----------
    t : np.ndarray
        Time array
    A : float
        General amplitude
    ab : float
        Breath amplitude
    fb : float
        Breath frecuency
    ah : float
        Heart pulse amplitude
    fh : float
        Heart pulse frecuency

    Returns
    -------
    np.ndarray
        Array of signal samples
    """
    return A * (ab * np.sin(2*np.pi*fb*t) + ah * np.sin(2*np.pi*fh*t))

def pulsesSig(t: np.ndarray, pw: float=2e-9, prf: float=7.85e6) -> np.ndarray:
    """Pulse signal

    Parameters
    ----------
    t : np.ndarray
        Time array
    pw : float, optional
        Pulse width, by default 2e-9
    prf : float, optional
        Pulse repetition frequency, by default 7.85e6

    Returns
    -------
    np.ndarray
        Array of signal samples
    """
    tScan = 1/prf
    scanLen = int(tScan / t[1])
    nScans = int(t.max() / tScan)
    tScans = t[:(nScans * scanLen)].reshape((nScans, scanLen))
    pulseStopTimes = tScans[:, int(pw / t[1])]
    return (tScans < pulseStopTimes[:, None]).flatten()

def pulsePosModSig(t: np.ndarray, pw: float=2e-9, prf: float=7.85e6,
                   m:int=19) -> np.ndarray:
    """Pulse position modulated signal

    Parameters
    ----------
    t : np.ndarray
        Time array
    pw : float, optional
        Pulse width, by default 2e-9
    prf : float, optional
        Pulse repetition frequency, by default 7.85e6
    m : float, optional
        Modulation index, by default 19

    Returns
    -------
    np.ndarray
        Array of signal samples
    """
    tScan = 1/prf
    scanLen = int(tScan / t[1])
    nScans = int(t.max() / tScan)
    tScans = t[:(nScans * scanLen)].reshape((nScans, scanLen))
    tScans = tScans - tScans[:,0][:,None]
    mod = t[1] * np.sin(2 * np.pi * t * 2 / (t[1] * len(t)))
    tStartInit = tScans[:, int(scanLen / 2)]
    tStart = tStartInit + mod[int(scanLen / 2): m * scanLen:scanLen]
    tStop = tStart + pw
    return np.logical_and(tScans < tStop[:, None],
                          tScans > tStart[:,None]).flatten()


def testSine(t: np.ndarray, f: float) -> Tuple[Tuple[np.ndarray], np.ndarray]:
    """Sine signal

    Parameters
    ----------
    t : np.ndarray
        Time array
    f : float
        Frequency

    Returns
    -------
    Tuple[Tuple[np.ndarray], np.ndarray]
        Instantaneous frequency and signal samples respectively
    """
    return (f*np.ones_like(t),), np.sin(2*np.pi*f*t)

def tritone(t: np.ndarray, f1: float, f2: float, f3: float
           )-> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Tritone signal

    Parameters
    ----------
    t : np.ndarray
        Time array
    f1 : float
        First tone frequency
    f2 : float
        Second tone frequency
    f3 : float
        Third tone frequency

    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        Tuple with instantaneous frequencies and signal samples respectively
    """

    signal = np.sin(2*np.pi*f1*t) + 0.5 * np.sin(2*np.pi*f2*t) + 0.2 * np.sin(2*np.pi*f3*t)
    f = (f1*np.ones_like(signal), f2*np.ones_like(signal), f3*np.ones_like(signal))
    return f, signal

def testChirp(t: np.ndarray, fmin: float, fmax: float) -> Tuple[Tuple[np.ndarray], np.ndarray]:
    """Chirp

    Parameters
    ----------
    t : np.ndarray
        Time array
    fmin : float
        Initial frequency
    fmax : float
        Final frequency

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of instantaneous frequencies and signal samples respectively
    """
    f = np.linspace(fmin, fmax, len(t))
    return (f,), spchirp(t, fmin, t[-1], fmax)  #np.sin(2*np.pi*f*t)

def testUpDownChirp(t: np.ndarray, fmin: float, fmax: float) -> Tuple[Tuple[np.ndarray], np.ndarray]:
    """Chirp signal going from fmin to fmax at time T/2, and returning to fmin at time T.

    Parameters
    ----------
    t : np.ndarray
        Time array
    fmin : float
        Initial frequency
    fmax : float
        T/2 frequency

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of instantaneous frequencies and signal samples respectively
    """
    f = np.linspace(fmin, fmax, int(len(t)/2))
    fr = np.concatenate((f,f[::-1]))
    return (f,), np.sin(np.pi*fr*t)

def quadraticChirp(t: np.ndarray, fmin: float, fmax: float) -> Tuple[Tuple[np.ndarray], np.ndarray]:
    """Quadratic chirp with vertex at T/2

    Parameters
    ----------
    t : np.ndarray
        Time array
    fmin : float
        Initial frequency
    fmax : float
        T/2 frequency

    Returns
    -------
   Tuple[np.ndarray, np.ndarray]
        Arrays of instantaneous frequencies and signal samples respectively
    """
    f0 = fmin
    f1 = fmax
    t1 = t[int(len(t)/2)]
    f = f1 - (f1 - fmin) * (t1 - t)**2 / t1**2

    return (f,), spchirp(t, f0=f0, f1=f1, t1=t1, method='quadratic',
                      vertex_zero=False)


def dualQuadraticChirps(t: np.ndarray, start_and_vertex_freqs1: Tuple,
                        start_and_vertex_freqs2: Tuple
                       ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Dual quadratic chirp signals

    Parameters
    ----------
    t : np.ndarray
        Time array
    start_and_vertex_freqs1 : Tuple
        Tuple of initial frequencies for both chirps
    start_and_vertex_freqs2 : Tuple
        Tuple of T/2 frequencies for both chirps

    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]
        Tuple containing tuple of arrays of instantaneous frequencies and signal samples.
    """
    f1, signal1 = quadraticChirp(t, start_and_vertex_freqs1[0], start_and_vertex_freqs1[1])
    f2, signal2 = quadraticChirp(t, start_and_vertex_freqs2[0], start_and_vertex_freqs2[1])
    signal = signal1 + signal2
    f = (f1[0], f2[0])
    return f, signal


def crossChrips(t: np.ndarray, fmin: float, fmax: float, N: int) -> Tuple:
    """Dual crossing chirp signals

    Parameters
    ----------
    t : np.ndarray
        Time array
    fmin : float
        Minimum frequency of the chirps
    fmax : float
        Maximum frequency of the chirps
    N : int
        Number of chirps

    Returns
    -------
    Tuple
        Tuple with list of instantaneous frequencies array and signal samples
    """

    nPoints = len(t)
    f_init = np.linspace(fmin, fmax, N)
    f_end = np.linspace(fmax, fmin, N)
    freqs = np.zeros((N, nPoints))
    chirps = np.zeros((N, nPoints))
    for i in range(N):
        freqs[i,:] = f_init[i] + (f_end[i] - f_init[i]) * (t / t[-1])
        chirps[i,:] = spchirp(t, f_init[i], t[-1], f_end[i])
    signal = chirps.sum(axis=0)
    return freqs, signal


def delta(t: np.ndarray, td: float) -> np.ndarray:
    """Delta in td

    Parameters
    ----------
    t : np.ndarray
        Time array
    td : float
        Delta location in time

    Returns
    -------
    np.ndarray
        Signal samples
    """
    sig = np.zeros_like(t)
    sig[np.argmax(t>td)] = 1
    return sig
