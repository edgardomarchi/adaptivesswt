import timeit
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np

from adaptivesswt.configuration import Configuration
from adaptivesswt.sswt import sswt
from adaptivesswt.utils import signal_utils as generator


def test_complexity(stopSignalTime: float = 20) -> np.ndarray:

    stopTime = stopSignalTime
    fs = 2000
    signalLen = int(stopTime * fs)
    t, ts = np.linspace(0, stopTime, signalLen, endpoint=False, retstep=True)

    _, signal = generator.quadraticChirp(t, 3, 7)

    config = Configuration(
        minFreq=1,
        maxFreq=10,
        numFreqs=32,
        ts=1 / 2000,
        wcf=0.5,
        wbw=ts,
        waveletBounds=(-3, 3),
        threshold=signal.max() / (100),
        numProc=24,
        plotFilt=False,
    )

    timingsProc = np.empty(cpu_count())
    passes = 4

    for i in np.arange(cpu_count()) + 1:
        config.numProc = i
        sswt_fix = lambda: sswt(signal, **config.asdict())
        timingsProc[i - 1] = (
            timeit.timeit(sswt_fix, number=passes) / passes / stopTime / fs
        )
        print(f'Number of processes : {i}')
        print(f'Average timing normalized over signal length = {timingsProc[i-1]} s/S')
        print(f'Maximum fs for true-real time {1/timingsProc[i-1]}')

    return timingsProc


if __name__ == '__main__':
    import platform

    # Uncomment if you have pyqt installed:
    # import matplotlib
    # matplotlib.use('Qt5Agg')

    numLengths = 10
    signalLengths = np.linspace(20, 200, numLengths, dtype=int)
    print(f'Performing test for signal lengths of {signalLengths} s.')
    timings = []
    for len in signalLengths:
        print(f'Signal length of {len}s')
        timings.append(test_complexity(len))
        print('--------------------', end='\n')

    plt.figure('Normalized time vs number of processes')
    plt.gca().set_xlabel('processes')
    plt.gca().set_ylabel('s/S',loc='top')
    for i in range(numLengths):
        plt.plot(
            np.arange(cpu_count()) + 1, timings[i], label=f'{signalLengths[i]} s signal'
        )
    plt.legend()
    plt.show()
