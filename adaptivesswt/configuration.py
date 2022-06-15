#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger()

import multiprocessing as mp
from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple, Union

import numpy as np
import pywt


@dataclass(init=True)
class Configuration:
    """SSWT Configuration helper class

    This dataclass contains as attributes all the parameters needed to correctly
    configure the SSWT function. It has common parameters by default.
    It also provides an `asdict()` method to pass those parameters as **kwargs

    Returns
    -------
    Configuration object
        Configuration object with attributes to use as parameters to SSWT

    Example
    -------
    >>> import adaptivesswt as asswt
    >>> config = asswt.Configuration(minFreq=1, maxFreq=10, numFreqs= 20, ts=(1/40))
    >>> ...
    >>> sst, freqs, tail = asswt.sswt(signal, **config.asdict())

    """

    minFreq: float
    maxFreq: float
    numFreqs: int
    ts: float = 1
    wcf: float = 1
    wbw: float = 1.5
    waveletBounds: Tuple[int, ...] = (-6, 6)
    threshold: float = 0.01
    numProc: int = mp.cpu_count()
    custom_scales: Union[np.ndarray, None] = None
    pad: int = 0
    plotFilt: bool = False
    wav: Optional[pywt.ContinuousWavelet] = None
    int_psi: Optional[np.ndarray] = None
    int_step: Optional[float] = None
    C_psi: complex = 1

    def __post_init__(self):
        self.wav = pywt.ContinuousWavelet(f'cmor{self.wbw}-{self.wcf}')
        self.wav.lower_bound, self.wav.upper_bound = self.waveletBounds
        self.int_psi, x = pywt.integrate_wavelet(self.wav)
        self.C_psi: complex = np.pi * np.conjugate(self.int_psi[np.argmin(np.abs(x))])  # type: ignore # Optional types will allways exist

        logger.info('Configuration created: %s\n', self)

    def asdict(self):
        return dict((field.name, getattr(self, field.name)) for field in fields(self))


if __name__ == '__main__':

    config = Configuration(minFreq=1, maxFreq=10, numFreqs=10)
    print(config)
    print(config.wav)
    print(config.asdict())
