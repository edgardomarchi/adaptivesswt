import numpy as np
from numba import njit, prange


@njit('float64[:], float64[:], float64[:], float64[:,:], complex128[:,:], complex128[:,:]',
      cache=True, parallel=True, fastmath=True)
def _freq_agregate_nb(deltaFreqs: np.ndarray, borderFreqs: np.ndarray,
                      aScale: np.ndarray, wab: np.ndarray, tr_matr: np.ndarray,
                      sst: np.ndarray):
    for b in prange(sst.shape[1]):      # type: ignore  # Time
        for w in prange(sst.shape[0]):  # type: ignore  # Frequency
            components = np.logical_and(wab[:,b] > borderFreqs[w],
                                        wab[:,b] <= borderFreqs[w+1])
            sst[w,b] = (tr_matr[components,b] * aScale[components]).sum() / deltaFreqs[w]
    return sst

@njit('float64[:], float64[:], float64[:], float64[:,:], complex128[:,:], complex128[:,:]',
      cache=True, parallel=True, fastmath=True)
def _freq_extract_nb(deltaFreqs: np.ndarray, borderFreqs: np.ndarray,
                     aScale: np.ndarray, wab: np.ndarray, tr_matr: np.ndarray,
                     set: np.ndarray)-> np.ndarray:
    for b in prange(set.shape[1]):        # Time
        for w in prange(set.shape[0]):    # Frequency
            if (wab[w,b] > borderFreqs[w]) and (wab[w,b] <= borderFreqs[w+1]):
                set[w,b] = tr_matr[w,b]  #/ deltaFreqs[w]
    return set

@njit('float64[:], float64[:,:], complex128[:,:], complex128[:,:]',
      cache=True, parallel=True, fastmath=True)
def _time_agregate_nb(time: np.ndarray, tab: np.ndarray,
                      tr_matr: np.ndarray, tsst: np.ndarray)-> np.ndarray:
    for w in prange(tsst.shape[0]):        # Frequency
        for b in prange(tsst.shape[1]):    # Time
            components = np.logical_and(tab[w,:] > time[b],
                                        tab[w,:] <= time[b+1])
            tsst[w,b] = (tr_matr[w,components]).sum() # / (2*np.pi)
    return tsst
