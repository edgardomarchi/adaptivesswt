import json
import logging
import os
from typing import Literal

# Logger:
logger = logging.getLogger(__name__)

# Init config:
__backendConfigFileName = 'backend.json'
__backendConfigFilePath = os.path.join(os.path.dirname(__file__),
                                       __backendConfigFileName)

# Load config:
__backendConfig:dict = {
    'backend': 'numba',
    'device_id': 0
}  # Default values
try:
    with open(__backendConfigFilePath, 'r', encoding='utf-8') as f:
        __backendConfig = json.load(f)
except json.JSONDecodeError as e:
    logger.warning('No valid json format! - Exception: %s', e)
except IOError:
    logger.warning('No backend found!')

logger.info('Using %s backend with device_id %d',
            __backendConfig['backend'], __backendConfig['device_id'])

############################
# Configuration functions: #
############################
def __setBackend(backend: Literal['numba','opencl','multiprocessing']):
    """Set the backend to use for computing

    It can only be set through command line arguments and its persistent.

    Parameters
    ----------
    backend : Literal["numba","opencl","multiprocessing"]
        The backend to use for computing. Default is "numba".
    """
    __backendConfig['backend'] = backend
    __saveBackendConfig()

def __setDeviceId(device_id: int):
    """Set the device to use for computing based on its id

    Parameters
    ----------
    device_id : int
        Device id to use with the selected backend
    """
    __backendConfig['device_id'] = device_id
    __saveBackendConfig()

def getBackend() -> str:
    """Return the current backend used for computing

    Returns
    -------
    str
        Name of the backend used for computing
    """
    return __backendConfig.get('backend', 'numba')  #type: ignore #since it is a string

def getDeviceId() -> int:
    """Return the configured device id from current backend

    Returns
    -------
    int
        Device id to use with the selected backend
    """
    return __backendConfig.get('device_id', 0)  #type: ignore #since it is an int

def __saveBackendConfig():
    """Saves the current backend configuration
    """
    with open(__backendConfigFilePath, 'w', encoding='utf-8') as f:
        json.dump(__backendConfig, f)


from .adaptivesswt import (
    adaptive_sswt,
    adaptive_sswt_overlapAndAdd,
    adaptive_sswt_slidingWindow,
)
from .configuration import Configuration
from .sswt import sswt
