#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import chirp as spchirp
from typing import Tuple

def testSig(t :np.ndarray) -> np.ndarray:
    """Señal de prueba como en "The Synchrosqueezing Transform: A EMD like tool" - Deaubechies.

    Parameters
    ----------
    t : np.ndarray
        Array de tiempos

    Returns
    -------
    np.ndarray
        Muestras de la señal
    """
    ventana1 = t < (5/2)*np.pi
    s1 = (0.*t + np.cos(2*np.pi*2*t)) * ventana1
    ventana2 = t > 2*np.pi
    s2 = np.cos((2/3)*((t-10)**3 - (2*np.pi - 10)**3)
                + 10*(t - 2*np.pi)) * ventana2
    return s1 + s2

def hbSig(t :np.ndarray, A: float=1,
          ab: float=0.018, fb: float=0.2,
          ah: float=0.004, fh: float=1) -> np.ndarray:
    """Señal armónica con dos componentes similar a respiración + pulso cardíaco

    Parameters
    ----------
    t : np.ndarray
        Array de tiempos
    A : float
        Amplitud general
    ab : float
        Amplitud respiración
    fb : float
        Frecuencia respiración
    ah : float
        Amplitud pulso cardíaco
    fh : float
        Frecuencia pulso cardíaco

    Returns
    -------
    np.ndarray
        Muestras de la señal        
    """
    return A * (ab * np.sin(2*np.pi*fb*t) + ah * np.sin(2*np.pi*fh*t))

def pulsesSig(t: np.ndarray, pw: float=2e-9, prf: float=7.85e6) -> np.ndarray:
    """Señal pulsante

    Parameters
    ----------
    t : np.ndarray
        Array de tiempos
    pw : float, optional
        Ancho de pulso, by default 2e-9
    prf : float, optional
        Frecuencia de repetición de pulso, by default 7.85e6

    Returns
    -------
    np.ndarray
        Muestras de la señal
    """
    tScan = 1/prf
    scanLen = int(tScan / t[1])
    nScans = int(t.max() / tScan)
    tScans = t[:(nScans * scanLen)].reshape((nScans, scanLen))
    pulseStopTimes = tScans[:, int(pw / t[1])] 
    return (tScans < pulseStopTimes[:, None]).flatten()

def pulsePosModSig(t: np.ndarray, pw: float=2e-9, prf: float=7.85e6,
                   m:float=19) -> np.ndarray:
    """Señal pulsante modulada en posición con una senoidal

    Parameters
    ----------
    t : np.ndarray
        Array de tiempos
    pw : float, optional
        Ancho de pulso, by default 2e-9
    prf : float, optional
        Frecuencia de repetición de pulso, by default 7.85e6
    m : float, optional
        Índice de modulación, by default 19

    Returns
    -------
    np.ndarray
        Mustras de la señal
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


def testSine(t: np.ndarray, f: float) -> np.ndarray:
    """Señal senoidal

    Parameters
    ----------
    t : np.ndarray
        Array de tiempos
    f : float
        Frecuencia

    Returns
    -------
    np.ndarray
        Muestras de la señal
    """
    return np.sin(2*np.pi*f*t)

def testChirp(t: np.ndarray, fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Señal chirp

    Parameters
    ----------
    t : np.ndarray
        Array de tiempos
    fmin : float
        Frecuencia inicial
    fmax : float
        Frecuencia final

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tupla con frecuencias instantáneas y muestras de la señal
    """
    f = np.linspace(fmin, fmax, len(t))
    return f, np.sin(np.pi*f*t)

def testUpDownChirp(t: np.ndarray, fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Señal chirp que alcanza fmax en T/2 y vuelve a fmin en T

    Parameters
    ----------
    t : np.ndarray
        Array de tiempos
    fmin : float
        Frecuencia inicial
    fmax : float
        Frecuencia final

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tupla con frecuencias instantáneas y muestras de la señal
    """
    f = np.linspace(fmin, fmax, int(len(t)/2))
    fr = np.concatenate((f,f[::-1]))
    return f, np.sin(np.pi*fr*t)

def quadraticChirp(t: np.ndarray, fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Chirp cuadrática con el vértice en T/2

    Parameters
    ----------
    t : np.ndarray
        Array de tiempos
    fmin : float
        Frecuencia mínima
    fmax : float
        Frecuencia máxima

    Returns
    -------
   Tuple[np.ndarray, np.ndarray]
        Tupla con frecuencias instantáneas y muestras de la señal
    """
    f0 = fmin
    f1 = fmax
    t1 = t[int(len(t)/2)]
    f = f1 - (f1 - fmin) * (t1 - t)**2 / t1**2

    return f, spchirp(t, f0=f0, f1=f1, t1=t1, method='quadratic',
                      vertex_zero=False)

def crossChrips(t: np.ndarray, fmin: float, fmax: float, N: int) -> Tuple:
    
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
    """Delta ubicada en td

    Parameters
    ----------
    t : np.ndarray
        Array de tiempos
    td : float
        Posición en tiempo de la delta

    Returns
    -------
    np.ndarray
        Mustras de la señal
    """
    sig = np.zeros_like(t)
    sig[np.argmax(t>td)] = 1
    return sig