#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger()

import numpy as np
import pywt
import multiprocessing as mp
from dataclasses import dataclass, asdict, fields
from typing import Tuple, Optional, Union

@dataclass(init=True)
class Configuration:
    minFreq: float
    maxFreq: float
    numFreqs: int
    ts: float = 1
    wcf: float = 1
    wbw: float = 1.5
    waveletBounds: Tuple[int,...] = (-6,6)
    threshold: float = 0.01
    numProc: int = mp.cpu_count()
    custom_scales: Union[np.ndarray, None] = None
    pad: int = 0
    plotFilt: bool = False
    wav: Optional[pywt.ContinuousWavelet] = None
    int_psi: Optional[np.ndarray] = None
    int_step: Optional[float] = None
    C_psi: Optional[complex] = None

    def __post_init__(self):
        self.wav = pywt.ContinuousWavelet(f'cmor{self.wbw}-{self.wcf}')
        self.wav.lower_bound, self.wav.upper_bound = self.waveletBounds
        self.int_psi, x = pywt.integrate_wavelet(self.wav)
        #self.int_step = x[1]-x[0]
        self.C_psi = np.pi * np.conjugate(self.int_psi[np.argmin(np.abs(x))])

        logger.info('Configuration created: %s\n',self)

    def asdict(self):
        return dict((field.name, getattr(self, field.name)) for field in fields(self))


if __name__ == '__main__':

    config = Configuration(minFreq=1, maxFreq=10, numFreqs=10)
    print(config)
    print(config.wav)
    print(config.asdict())
