from imp import reload
from typing import Literal

__backend = 'numba'

def setBackend(backend: Literal['numba','opencl','multiprocessing']):
    __backend = backend

def getBackend() -> str:
    return __backend

from .adaptivesswt import (
    adaptive_sswt,
    adaptive_sswt_overlapAndAdd,
    adaptive_sswt_slidingWindow,
    main,
)
from .configuration import Configuration
from .sswt import sswt
