#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Tuple

@dataclass
class Configuration:
    minFreq: float
    maxFreq: float
    numFreqs: int
    ts: float = 1
    wcf: float = 1
    wbw: float = 1.5
    waveletBounds: Tuple[int,...] = (-8,8)
    umbral: float = 0.01
    numProc: int = mp.cpu_count()
    custom_scales: np.ndarray = None
    log: bool = False
    plotFilt: bool = False

    def asdict(self):
        return asdict(self)