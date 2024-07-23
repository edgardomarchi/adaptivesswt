import argparse
import json
import logging
import os
from typing import Literal

# Logger:
logger = logging.getLogger(__name__)

# Init config:
__backendConfigFileName = 'backend.json'
__backendConfigFilePath = os.path.join(os.path.dirname(__file__), __backendConfigFileName)

__defaultBackendConfig = {
    'backend': 'numba',
    'device_id': 0
}

__backendConfig = {}
try:
    with open(__backendConfigFilePath, 'r') as plFile:
        __backendConfig= json.load(plFile)
except json.JSONDecodeError as e:
    __backendConfig = __defaultBackendConfig
    logger.warning('No valid json format! - Exception: %s', e)
except IOError:
    __backendConfig = __defaultBackendConfig
    logger.warning('No registered plugins list found!')

logger.info('Using %s backend with device_id %d',
            __backendConfig['backend'], __backendConfig['device_id'])

def setBackend(backend: Literal['numba','opencl','multiprocessing']):
    __backendConfig['backend'] = backend
    __saveBackendConfig()

def setDeviceId(device_id: int):
    __backendConfig['device_id'] = device_id
    __saveBackendConfig()

def getBackend() -> str:
    return __backendConfig['backend']

def __saveBackendConfig():
    with open(__backendConfigFilePath, 'w') as plFile:
        json.dump(__backendConfig, plFile)


from .adaptivesswt import (
    adaptive_sswt,
    adaptive_sswt_overlapAndAdd,
    adaptive_sswt_slidingWindow,
)
from .configuration import Configuration
from .sswt import sswt


def main():
    parser = argparse.ArgumentParser(description='Configuration for Adaptive SSWT')

    parser.add_argument('-b', '--backend', type=str, choices=['numba','opencl','multiprocessing'], default='numba',
                        help='Backend to use: numba, opencl, multiprocessing')
    parser.add_argument('-d', '--device_id', type=int, default=0,
                        help='Device id to use with computing backend')

    parser.add_argument('-i', '--info', help='Show current configuration', action='store_true')

    args = parser.parse_args()
    if args.info:
        print('Current configuration:')
        print(f'Backend: {getBackend()}')
        print(f'Device id: {__backendConfig["device_id"]}')
        return

    setBackend(args.backend)
    setDeviceId(args.device_id)


if __name__ == '__main__':
    main()
