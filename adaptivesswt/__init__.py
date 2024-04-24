from typing import Literal

__backend = 'opencl'

def setBackend(backend: Literal['numba','opencl','multiprocessing']):
    __backend = backend

from .adaptivesswt import (
    adaptive_sswt,
    adaptive_sswt_overlapAndAdd,
    adaptive_sswt_slidingWindow,
    main,
)
from .configuration import Configuration
from .sswt import sswt
