import argparse
import logging

from adaptivesswt import __setBackend, __setDeviceId, getBackend, getDeviceId

logger = logging.getLogger(__name__)

def cli():
    parser = argparse.ArgumentParser(description='Configuration for Adaptive SSWT')

    parser.add_argument('-b', '--backend', type=str, choices=['numba','opencl','multiprocessing'], default='numba',
                        help='Backend to use: numba, opencl, multiprocessing')
    parser.add_argument('-d', '--device_id', type=int, default=0,
                        help='Device id to use with computing backend')

    parser.add_argument('-i', '--info', help='Show current configuration', action='store_true')

    parser.add_argument('-l', '--log', type=str, default='INFO', help='Logging level')

    args = parser.parse_args()
    if args.info:
        print('Current configuration:')
        print(f'Backend: {getBackend()}')
        print(f'Device id: {getDeviceId()}')
        return

    __setBackend(args.backend)
    __setDeviceId(args.device_id)


if __name__ == '__main__':
    # Logging configuration
    logging.basicConfig(filename='sswt_test.log', filemode='w',
                        format='%(levelname)s - %(asctime)s - %(name)s:\n %(message)s')
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    cli()
