from typing import Tuple, Union

import numpy as np
import pywt


def getScale(freq: Union[float, np.ndarray], ts: float, wcf: float) -> Union[float, np.ndarray]:
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
    return wcf / (ts * freq)


def calcScalesAndFreqs(ts: float, wcf: float, fmin: float, fmax: float,
                       nv: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates scales and frequencies for CWT within the interval [fmin, fmax) or [fmin, fmax]

    Parameters
    ----------
    fs : float
        Sampling frequency
    wcf : float
        Wavelet central frequency
    fmin : float
        Minimum frequency (included)
    fmax : float
        Maximum frequency (included if endpoint is True)
    nv : int
        Number of voices or frequencies

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns tuple with (scales, frequencies, delta scales, delta frequencies)
    """

    # Frequencies within interval (fmin, fmax)
    freqs = np.linspace(fmin, fmax, nv+2, endpoint=True)[1:-1]

    scales = getScale(freqs, ts, wcf)

    deltaFreqs, _ = getDeltaAndBorderFreqs(freqs)        # dF0, dF1, ... , dF<nv+1> -> From low to high frecuencies
    deltaScales = getDeltaScales(scales)  # type: ignore # da<nv+1>, da<nv>, ... , da0 -> From high to low scales
    return scales, freqs, deltaScales, deltaFreqs  # type: ignore  # safe since freqs is an array thus scales will be array

def getDeltaScales(scales: np.ndarray) -> np.ndarray:
    """Calculates delta Scales for SST

    Parameters
    ----------
    scales : np.ndarray
        Array with used scales in CWT, its assumed to be ordered from max to min.

    Returns
    -------
    np.ndarray
        Array of delta scales correspondig to 'scales'
    """
    if len(scales) > 1:
        deltaScales = -1 * np.diff(scales, append=scales[-2])  # last delta equals previous since no info
        deltaScales[-1] *= -1
    else:
        deltaScales = scales
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
    if len(freqs) > 1:
        deltaFreqs = np.diff(freqs, prepend=freqs[1])
        deltaFreqs[0] *= -1
        borderFreqs = np.concatenate((freqs-deltaFreqs/2,
                                      np.array([freqs[-1]+deltaFreqs[-1]/2])))
        borderFreqs = np.where(borderFreqs < 0, 0, borderFreqs)  # delta frequency could be larger than frequency
                                                                 # if the separation is relatively big, since border
                                                                 # frequency could be negative -> saturate to 0
        deltaFreqs = np.diff(borderFreqs)  # For non linear spacing this updates delta frequencies
    else:
        deltaFreqs, borderFreqs = freqs, freqs

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

    print(f'Scale for {f}Hz with fs={fs} and wcf={wcf}: {getScale(f,1/fs,wcf)}', end='\n')
    scales, freqs, deltaScales, deltaFreqs = calcScalesAndFreqs(1/fs, wcf, fmin, fmax, nv)
    print(f'Scales: {scales}')
    print(f'Frequencies: {freqs}')
    print(f'Delta Scales: {deltaScales}')
    print(f'Delta Frequencies: {deltaFreqs}')
    newDelta, newBorder = getDeltaAndBorderFreqs(freqs)
    print(f'New Delta Frequencies: {newDelta}')
    print(f'Border Frequencies: {newBorder}')
