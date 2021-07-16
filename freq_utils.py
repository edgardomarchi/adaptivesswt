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
