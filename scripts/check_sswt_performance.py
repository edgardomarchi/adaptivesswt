import timeit
from multiprocessing import cpu_count

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import adaptivesswt
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
        min_freq=1,
        max_freq=10,
        num_freqs=32,
        ts=1 / 2000,
        wcf=1,
        wbw=2,
        wavelet_bounds=(-8, 8),
        threshold=signal.max() / (100),
        plot_filters=False,
    )

    timingsProc = np.empty(cpu_count())
    passes = 4

    for i in np.arange(cpu_count()) + 1:
        config.num_processes = i
        sswt_fix = lambda: sswt(signal, **config.asdict())
        timingsProc[i - 1] = (
            timeit.timeit(sswt_fix, number=passes) / passes / stopTime / fs
        )
        print(f'Number of processes : {i}')
        print(f'Average timing normalized over signal length = {timingsProc[i-1]:.4E} s/S')
        print(f'Maximum fs for true-real time {1/timingsProc[i-1]:.4E}')

    return timingsProc


def ckeck_complexity_distribution(
    maxSignalTime: float = 20,
    n_steps: int = 10,
    itl: bool = True,
    method: str = 'proportional',
    maxIters: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    maxSignalTimes = np.linspace(
        maxSignalTime / n_steps, maxSignalTime, n_steps, endpoint=True
    )
    a_times = np.zeros_like(maxSignalTimes)
    sst_times = np.zeros_like(maxSignalTimes)
    asst_times = np.zeros_like(maxSignalTimes)

    config = Configuration(
        min_freq=1,
        max_freq=10,
        num_freqs=32,
        ts=1 / 2000,
        wcf=1,
        wbw=2,
        wavelet_bounds=(-8, 8),
        threshold=(1/10),
        plot_filters=False,
    )
    thrsh = (1/5)

    passes = 10

    for i, stopSignalTime in enumerate(maxSignalTimes):
        stopTime = stopSignalTime
        fs = 1 / config.ts
        signalLen = int(stopTime * fs)
        t, ts = np.linspace(0, stopTime, signalLen, endpoint=False, retstep=True)

        #_, signal = generator.tritone(t, 3, 4, 7)
        _, signal = generator.quadraticChirp(t, 3, 7)
        sswt_fix = lambda: sswt(signal, **config.asdict())
        sst_times[i] = timeit.timeit(sswt_fix, number=passes) / passes

        asswt_fix = lambda: adaptive_sswt(
            signal, maxIters, method, thrsh, itl, **config.asdict()
        )
        asst_times[i] = timeit.timeit(asswt_fix, number=passes) / passes

        print(f'INFO: Pass {i+1}/{n_steps}')

    # Adaptive stage runs maxIters times, meanwhile sst runs maxIters+1 times
    if maxIters == 0:
        a_times = asst_times - sst_times  # It should be zero, but runtimes are not exact
    else:
        a_times = (asst_times - sst_times * (maxIters + 1)) / maxIters

    return maxSignalTimes, a_times, sst_times, asst_times


def main():

    plt.rcParams['text.usetex'] = True
    font = {'family': 'normal', 'weight': 'normal', 'size': 10}
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

        fig, ax = plt.subplots(1, figsize=(10/2,10/2))
        # fig.suptitle('Max sampling frequency vs number of processes')
        fig.set_tight_layout(True)
        ax.set_xlabel(r'\# proc.')
        ax.set_ylabel(r'$fs_{MAX}\,[Hz]$', loc='top')
        for i in range(numLengths):
            plt.plot(
                np.arange(cpu_count()) + 1,
                1/timings[i],
                label=r'$t_{MAX}=$'+f'{signalLengths[i]}s',
            )
        plt.legend()

    else:
        fig, ax = plt.subplots(1, figsize=(16/2,9/2))
        fig.set_tight_layout(True)
        maxSignalTime = 80
        steps=3
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        for i, maxiters in enumerate([0, 1, 2, 4, 6]):
            print(f'\nPerforming test for maxIters={maxiters}:')
            s_len, a_times, sst_times, asst_times = ckeck_complexity_distribution(
                maxSignalTime=maxSignalTime,
                n_steps=steps,
                maxIters=maxiters
            )
            ax.plot(s_len, (maxiters + 1)*sst_times, f'{colors[i]}--', label='SST' r'$\times$' f'[(Iter.={maxiters})+1]')
            ax.plot(s_len, asst_times, colors[i], label=f'ASST (Iter.={maxiters})')

        #ax.plot(s_len, a_times, 'b', label='Single Adaptive stage')

        ax.set_xlabel(r'$t_{MAX}\,[s]$')
        ax.set_ylabel('Run-time [s]')
        ax.set_title(f'Backend: {adaptivesswt.getBackend()}')

        ax.legend()

    plt.show()


if __name__ == '__main__':
    main()
