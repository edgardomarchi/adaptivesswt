import timeit
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np

from adaptivesswt import adaptive_sswt
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
        ts=1/2000,
        wcf=1,
        wbw=2,
        waveletBounds=(-8, 8),
        threshold=signal.max() / (100),
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

def ckeck_complexity_distribution(maxSignalTime: float = 20, n_steps: int=10,
                                  itl:bool = True, method:str = 'proportional', maxIters:int = 4,
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    maxSignalTimes = np.linspace(maxSignalTime/n_steps, maxSignalTime, n_steps, endpoint=True)
    a_times = np.zeros_like(maxSignalTimes)
    sst_times = np.zeros_like(maxSignalTimes)
    asst_times = np.zeros_like(maxSignalTimes)

    config = Configuration(
            minFreq=1,
            maxFreq=10,
            numFreqs=32,
            ts=1/2000,
            wcf=1,
            wbw=2,
            waveletBounds=(-8, 8),
            threshold=1/10,
            numProc=24,
            plotFilt=False,
        )
    thrsh = 1/5

    passes = 4

    for i, stopSignalTime in enumerate(maxSignalTimes):
        stopTime = stopSignalTime
        fs = 1/config.ts
        signalLen = int(stopTime * fs)
        t, ts = np.linspace(0, stopTime, signalLen, endpoint=False, retstep=True)

        _, signal = generator.quadraticChirp(t, 3, 7)

        sswt_fix = lambda: sswt(signal, **config.asdict())
        sst_times[i] = timeit.timeit(sswt_fix, number=passes) / passes


        asswt_fix = lambda: adaptive_sswt(signal, maxIters, method, thrsh, itl,**config.asdict())
        asst_times[i] = timeit.timeit(asswt_fix, number=passes) / passes

        print(f'INFO: Pass {i+1}/{n_steps}')

    # Adaptive stage runs maxIters times, menwhile sst runs maxIters+1 times
    # (e.g. maxIters = 0 runs the non-adaptive algorithm, i.e. sst only):
    a_times = asst_times/maxIters - sst_times*(maxIters+1)/maxIters

    return maxSignalTimes, a_times, sst_times, asst_times


if __name__ == '__main__':
    import matplotlib
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 10}

    matplotlib.rc('font', **font)

    # Uncomment if you have pyqt installed:
    # import matplotlib
    # matplotlib.use('Qt5Agg')

    print("Press 'p' for paralellism test, any other key for timing distribution")
    sel = input()

    if sel == 'p':
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

    else:
        s_len, a_times, sst_times, asst_times = ckeck_complexity_distribution(50, maxIters=2)
        fig, ax = plt.subplots(1, dpi=300)
        ax.plot(s_len, a_times, 'b', label='Single Adaptive stage')
        ax.plot(s_len, sst_times, 'b:', label='Single SSWT')
        ax.plot(s_len, asst_times, 'b--', label='Full ASSWT')

        s_len, a_times, sst_times, asst_times = ckeck_complexity_distribution(50, maxIters=4)
        ax.plot(s_len, a_times, 'r', label='Single Adaptive stage')
        ax.plot(s_len, sst_times, 'r:', label='Single SSWT')
        ax.plot(s_len, asst_times, 'r--', label='Full ASSWT')

        ax.set_xlabel('Signal duration [s]')
        ax.set_ylabel('Run time [s]')

        ax.legend()

    plt.show()
