#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from dataclasses import dataclass, fields
from multiprocessing import cpu_count
from typing import Optional, Tuple, Union

import numpy as np
from pywt import (  # type: ignore # Pylance seems to fail finding ContinuousWavelet within pywt
    ContinuousWavelet,
    integrate_wavelet,
)

logger = logging.getLogger()


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
    >>> config = asswt.Configuration(min_freq=1, max_freq=10, num_freqs= 20, ts=(1/40))
    >>> ...
    >>> sst, freqs, tail = asswt.sswt(signal, **config.asdict())

    """

    min_freq: float
    max_freq: float
    num_freqs: int
    ts: float = 1
    wcf: float = 1
    wbw: float = 1.5
    wavelet_bounds: Tuple[int, ...] = (-6, 6)
    threshold: float = 0.01
    num_processes: int = cpu_count()
    custom_scales: Union[np.ndarray, None] = None
    pad: int = 0
    plot_filters: bool = False
    wav: Optional[ContinuousWavelet] = None
    int_psi: Optional[np.ndarray] = None
    int_step: Optional[float] = None
    c_psi: complex = 1
    transform: str = 'sst' # 'sst', 'tsst', 'tfr'

    def __post_init__(self):
        self.wav = ContinuousWavelet(f'cmor{self.wbw}-{self.wcf}')
        self.wav.lower_bound, self.wav.upper_bound = self.wavelet_bounds
        self.int_psi, x = integrate_wavelet(self.wav)  # type: ignore # integrate wavelet always operates with a ContinuousWavelet object
        self.c_psi: complex = np.pi * np.conjugate(self.int_psi[np.argmin(np.abs(x))])  # type: ignore # Optional types will allways exist

        logger.info('Configuration created: %s\n', self)

    def asdict(self):
        """ Returns the configuration as a dictionary.

        Returns
        -------
        dict
            Dictionary containing configuration parameters.
        """
        return dict((field.name, getattr(self, field.name)) for field in fields(self))


def main():
    """ Main function for testing.
    """
    config = Configuration(min_freq=1, max_freq=10, num_freqs=10)
    print(config)
    print(config.wav)
    print(config.asdict())

if __name__ == '__main__':
    main()
