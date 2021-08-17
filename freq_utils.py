import numpy as np
from typing import Tuple


def getScale(freq: float, ts: float, wcf: float) -> float:
    """Devuelve la escala (a) asociada a la frecuencia f

    Parameters
    ----------
    freq : float
        Frecuencia a obtener la escala
    ts : float
        Tiempo de muestreo
    wcf : float
        Frecuencia central de la wavelet

    Returns
    -------
    float
        Escala correspondiente a la frecuencia freq
    """    
    return 1 / ((ts / wcf) * freq)


def calcScalesAndFreqs(ts: float, wcf: float, fmin: float, fmax: float, nv: float,
                       log: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcula las escalas y las frecuencias para la CWT dentro del intervalo abierto (fmin, fmax)

    Parameters
    ----------
    fs : float
        Frecuencia de muestreo
    wcf : float
        Frecuencia central de la Wavelet
    fmin : float
        Frecuencia mínima del intervalo (no incluída)
    fmax : float
        Frecuencia máxima del intervalo (no incluída)
    nv : float
        Cantidad de voces o frecuencias
    log : bool, optional
        Determina si la separación entre frecuencias es lineal o logarítmica, by default False

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Devuelve una tupla con (escalas, frecuencias, delta escalas, delta frecuencias)
    """

    # Frequencies within open interval (fmin, fmax)
    if log:
        freqs = np.logspace(np.log10(fmin), np.log10(fmax), nv+2)
    else:
        freqs = np.linspace(fmin, fmax, nv+2)

    getScales = lambda freqs : getScale(freqs, ts, wcf)

    scales = getScales(freqs)

    deltaFreqs = np.diff(freqs)[:-1]  # dF0, dF1, ... , dF<nv+1> -> De frecuencias chicas a grandes
    deltaScales = -1 * np.diff(scales)[1:]  # da<nv+1>, da<nv>, ... , da0 -> De escalas grandes a chicas

    return scales[1:-1], freqs[1:-1], deltaScales, deltaFreqs


def getDeltaAndBorderFreqs(freqs: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    """Devuelve los delta de frecuencia y las frecuencias borde de los bins

    Parameters
    ----------
    freqs : np.ndarray
        Array de frecuencias centrales

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tupla con los deltaF y las frecuencias borde
    """
    deltaFreqs = np.diff(freqs, prepend=freqs[1])
    deltaFreqs[0] *= -1
    borderFreqs = np.concatenate((freqs-deltaFreqs/2,
                                  np.array([freqs[-1]+deltaFreqs[-1]/2])))
    deltaFreqs = np.diff(borderFreqs)
    return deltaFreqs, borderFreqs


if __name__=='__main__':

    fs = 10
    wcf = 0.5
    f =  2.5
    print(f'Escala para {f}Hz con fs={fs} y wcf={wcf}: {getScale(f,1/fs,wcf)}', end='\n')

    scales, freqs, deltaScales, deltaFreqs = calcScalesAndFreqs(1/fs, wcf, 0.5, 5, 10, log=True)
    print(f'Escalas: {scales}')
    print(f'Frecuencias: {freqs}')
    print(f'Delta Escalas: {deltaScales}')
    print(f'Delta Frecuencias: {deltaFreqs}')
