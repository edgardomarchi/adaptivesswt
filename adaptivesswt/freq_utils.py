import pywt
import numpy as np
from typing import Tuple


def getScale(freq: float, ts: float, wcf: float) -> float:
    """Returns 'a' scale asociated to 'f' frequency

    Parameters
    ----------
    freq : float
        Frequency to get scale
    ts : float
        Sample time
    wcf : float
        Wavelet central frequency

    Returns
    -------
    float
        Scale corresponding to frequency 'freq'
    """    
    return wcf / (ts * freq)  #  1 / ((ts / wcf) * freq)


def calcScalesAndFreqs(ts: float, wcf: float, fmin: float, fmax: float, nv: float,
                       log: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates scales and frequencies for CWT within the closed interval [fmin, fmax]

    Parameters
    ----------
    fs : float
        Sampling frequency
    wcf : float
        Wavelet central frequency
    fmin : float
        Minimum frequency (included)
    fmax : float
        Maximum frequency (included)
    nv : float
        Number of voices or frequencies
    log : bool, optional
        True if frequency spacing is logarithmic; linear if False, by default False

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns tuple with (scales, frequencies, delta scales, delta frequencies)
    """

    # Frequencies within clsed interval [fmin, fmax]
    if log:
        freqs = np.logspace(np.log10(fmin), np.log10(fmax), nv)
    else:
        freqs = np.linspace(fmin, fmax, nv)

    getScales = lambda freqs : getScale(freqs, ts, wcf)

    scales = getScales(freqs)

    deltaFreqs, _ = getDeltaAndBorderFreqs(freqs) # dF0, dF1, ... , dF<nv+1> -> From low to high frecuencies
    deltaScales = getDeltaScales(scales)  # da<nv+1>, da<nv>, ... , da0 -> From high to low scales
    return scales, freqs, deltaScales, deltaFreqs

def getDeltaScales(scales: np.ndarray) -> np.ndarray:
    """Calculates delta Scales for SSWT

    Parameters
    ----------
    scales : np.ndarray
        Array with used scales in CWT, its assumed to be ordered from max to min.

    Returns
    -------
    np.ndarray
        Array of delta scales correspondig to 'scales'
    """
    deltaScales = -1 * np.diff(scales, append=scales[-2])  # last delta equals previous since no info
    deltaScales[-1] *= -1
    return deltaScales

def getDeltaAndBorderFreqs(freqs: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    """Calculates delta frequencies and bins border frequencies

    Parameters
    ----------
    freqs : np.ndarray
        Central frequencies

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple with delta frequecies and border frequencies
    """
    deltaFreqs = np.diff(freqs, prepend=freqs[1])
    deltaFreqs[0] *= -1
    borderFreqs = np.concatenate((freqs-deltaFreqs/2,
                                  np.array([freqs[-1]+deltaFreqs[-1]/2])))
    deltaFreqs = np.diff(borderFreqs)
    return deltaFreqs, borderFreqs

def calcFilterLength(wav: pywt.ContinuousWavelet, scale: float):
    """Calculates Wavelet filter length

    Parameters
    ----------
    wav : pywt.ContinuousWavelet
        Wavelet family object (PyWavelets)
    scale : float
        Scale to calculate filter length

    Returns
    -------
    int
        Wavelet filter length
    """
    width = wav.upper_bound - wav.lower_bound
    return int(np.ceil(scale * width + 1)) # See waveletExample.py

if __name__=='__main__':

    fs = 200
    wcf = 0.5
    f =  0.1
    fmin = 0.1
    fmax = 10
    nv = 10
    log = False

    print(f'Sacle for {f}Hz with fs={fs} and wcf={wcf}: {getScale(f,1/fs,wcf)}', end='\n')
    scales, freqs, deltaScales, deltaFreqs = calcScalesAndFreqs(1/fs, wcf, fmin, fmax, nv, log=log)
    print(f'Scales: {scales}')
    print(f'Frequencies: {freqs}')
    print(f'Delta Scales: {deltaScales}')
    print(f'Delta Frequencies: {deltaFreqs}')
