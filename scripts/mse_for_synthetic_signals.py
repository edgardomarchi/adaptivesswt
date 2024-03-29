from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from adaptivesswt.adaptivesswt import (
    adaptive_sswt,
    adaptive_sswt_overlapAndAdd,
    adaptive_sswt_slidingWindow,
)
from adaptivesswt.configuration import Configuration
from adaptivesswt.sswt import sswt
from adaptivesswt.utils.measures_utils import renyi_entropy
from adaptivesswt.utils.plot_utils import plot_batched_tf_repr, plot_tf_repr


def detect_frequencies(tr_matr, freqs, border=None):
    borderIdx: int = np.argmax(freqs > border) if border is not None else 0  # type: ignore
    try:
        f1 = freqs[np.argmax(abs(tr_matr[:borderIdx, :]), axis=0)]
    except ValueError:
        f1 = np.zeros(tr_matr.shape[1])

    f2 = freqs[np.argmax(abs(tr_matr[borderIdx:, :]), axis=0) + borderIdx]

    f = (f1, f2) if f1.any() else (f2, f1)
    return f


def get_mse_batched(
    signal: np.ndarray,
    t: np.ndarray,
    f: Tuple,
    transform: Callable,
    config: Configuration,
    method_fig: Optional[plt.figure] = None,
    if_fig: Optional[plt.figure] = None,
    **kwargs,
):
    border = None if len(f) == 1 else (config.max_freq + config.min_freq) / 2

    entropies = [0.0, 0.0, 0.0, 0.0]

    sst, cwt, freqs, _, _ = sswt(signal, **config.asdict())
    entropies[0] = renyi_entropy(cwt)
    entropies[1] = renyi_entropy(sst)
    if method_fig is not None:

        gs_mf = method_fig.add_gridspec(1, 4)
        cwt_ax = method_fig.add_subplot(
            gs_mf[0, 0],
        )
        sst_ax = method_fig.add_subplot(
            gs_mf[0, 1],
        )
        sst_ax.sharey(cwt_ax)
        asst_ax = method_fig.add_subplot(
            gs_mf[0, 2],
        )
        asst_ax.sharey(cwt_ax)
        basst_ax = method_fig.add_subplot(
            gs_mf[0, 3],
        )
        basst_ax.sharey(cwt_ax)
        plot_tf_repr(cwt, t, freqs, cwt_ax)
        cwt_ax.set_title('CWT')
        plot_tf_repr(sst, t, freqs, sst_ax)
        sst_ax.set_title('SST')

    f1_sst, f2_sst = detect_frequencies(np.abs(sst), freqs, border)
    f1_cwt, f2_cwt = detect_frequencies(np.abs(cwt), freqs, border)

    asst, aFreqs, _, _ = adaptive_sswt(
        signal,
        kwargs['bMaxIters'],
        kwargs['method'],
        kwargs['threshold'],
        kwargs['itl'],
        **config.asdict(),
    )

    entropies[2] = renyi_entropy(asst)
    if method_fig is not None:
        plot_tf_repr(asst, t, aFreqs, asst_ax)
        asst_ax.set_title('ASST')

    f1_asst, f2_asst = detect_frequencies(np.abs(asst), aFreqs, border)

    try:
        f[1]
    except IndexError:
        f = (f[0], np.zeros_like(f[0]))

    mse_cwt = (f[0] - f1_cwt) ** 2 + (f[1] - f2_cwt) ** 2
    mse_sst = (f[0] - f1_sst) ** 2 + (f[1] - f2_sst) ** 2
    mse_asst = (f[0] - f1_asst) ** 2 + (f[1] - f2_asst) ** 2

    mse_cwt_total = mse_cwt.sum() / len(mse_cwt)
    mse_sst_total = mse_sst.sum() / len(mse_sst)
    mse_asst_total = mse_asst.sum() / len(mse_asst)

    batchs = transform(
        kwargs['bLen'],
        signal,
        kwargs['bMaxIters'],
        kwargs['method'],
        kwargs['threshold'],
        kwargs['itl'],
        **config.asdict(),
    )

    f1BatchList = []
    f2BatchList = []
    entropiesBatchList = []
    # Recover instantaneous frequencies:
    for asswtb, freqsb, _, _ in batchs:
        entropiesBatchList.append(renyi_entropy(asswtb))
        f1b, f2b = detect_frequencies(asswtb, freqsb, border)
        f1BatchList.append(f1b)
        f2BatchList.append(f2b)

    entropies[3] = np.array(entropiesBatchList).sum() / len(batchs)

    f1_batch = np.array(f1BatchList[1:-1]).flatten()
    f1_batch = np.concatenate((f1BatchList[0], f1_batch, f1BatchList[-1]))
    f2_batch = np.array(f2BatchList[1:-1]).flatten()
    f2_batch = np.concatenate((f2BatchList[0], f2_batch, f2BatchList[-1]))

    mse_asst_batch = (f[0][: len(f1_batch)] - f1_batch) ** 2 + (
        f[1][: len(f2_batch)] - f2_batch
    ) ** 2
    mse_asst_batch_total = mse_asst_batch.sum() / len(mse_asst_batch)

    if method_fig is not None:
        plot_batched_tf_repr(batchs, config.ts, basst_ax)
        method_fig.set_tight_layout(True)

    ### TODO: This code is assuming two frequency components in the signal. It should be generalized.
    if if_fig is not None:

        gs_if = if_fig.add_gridspec(1, 2)
        if_comp_ax = if_fig.add_subplot(
            gs_if[0, 0],
        )
        mse_comp_ax = if_fig.add_subplot(
            gs_if[0, 1],
        )

        if_comp_ax.plot(t, f1_sst, ':', color='blue', label='SST')
        if_comp_ax.plot(t, f2_sst, ':', color='blue')

        if_comp_ax.plot(t, f1_batch, '-', alpha=0.9, color='red', label='B-ASST')
        if_comp_ax.plot(t, f2_batch, '-', alpha=0.9, color='red')

        for i, freq in enumerate(f):
            if i == 0:
                if_comp_ax.plot(
                    t[: len(signal)], freq, '--', color='green', label='Inst. Freq.'
                )
            else:
                if_comp_ax.plot(t[: len(signal)], freq, '--', color='green')
        if_comp_ax.set_title('(a) Instantaneous Frequencies')
        if_comp_ax.set_ylabel('frequency [Hz]', loc='top')
        if_comp_ax.legend()

        mse_comp_ax.plot(t[: len(mse_sst)], mse_sst, ':', color='blue', label='SST')
        mse_comp_ax.plot(
            t[: len(mse_asst_batch)], mse_asst_batch, '-', color='red', label='B-ASST'
        )
        mse_comp_ax.set_title('(b) MSE(t)')
        mse_comp_ax.set_xlabel('Time [s]', loc='right')
        mse_comp_ax.set_ylabel('MSE', loc='top')

        mse_comp_ax.legend()

        if_fig.set_tight_layout(True)

    return mse_cwt_total, mse_sst_total, mse_asst_total, mse_asst_batch_total, entropies


def main():
    import itertools
    from os.path import abspath, dirname
    from pathlib import Path

    import matplotlib
    import pandas as pd

    font = {'family': 'normal', 'weight': 'normal', 'size': 6}

    matplotlib.rc('font', **font)
    plt.rcParams['text.usetex'] = True

    # Uncomment if you have pyqt installed:
    # import matplotlib
    # matplotlib.use('Qt5Agg')

    from adaptivesswt.utils import signal_utils as generator

    parentDir = Path(dirname(dirname(abspath(__file__))))

    #################
    ## Setup test: ##
    #################
    stopTime = 12
    fs = 400
    signalLen = int(stopTime * fs)

    t, step = np.linspace(0, stopTime, signalLen, endpoint=False, retstep=True)
    ts = float(step)

    signals = {
        'Sine': generator.testSine(t, 30),
        'Linear Chirp': generator.testChirp(t, 25, 35),
        'Quadratic Chirp': generator.quadraticChirp(t, 28, 32),
        'Dual sine': generator.dualQuadraticChirps(t, (30, 30), (40, 40)),
        'Dual Quadratic Chirp': generator.dualQuadraticChirps(t, (28, 30), (42, 38)),
    }

    config = Configuration(
        min_freq=20,
        max_freq=50,
        num_freqs=16,
        ts=ts,
        wcf=1,
        wbw=30,
        wavelet_bounds=(-8, 8),
        threshold=1/10,
    )

    ## Configuration for batched version:
    #####################################
    num_batchs = 6  # Number of batches in the current frame
    batch_time = stopTime / num_batchs  # Duration of each batch in time
    bLen = int(
        len(t[t <= (num_batchs * batch_time)]) // num_batchs
    )  # Batch length in samples
    bPad = int(bLen * 0.9)  # Padding / overlapping per batch in samples
    config.pad = bPad
    threshold = config.threshold * 5  # Threshold for ASST
    bMaxIters = 2  # Max iterations in batched mode

    # Transform function to run
    tr = adaptive_sswt_slidingWindow

    # Results plaeceholder
    results = pd.DataFrame(
        data={
            'CWT': [0, 0, 0, 0, 0],
            'SST': [0, 0, 0, 0, 0],
            'ASST-ITL-prop': [0, 0, 0, 0, 0],
            'ASST-ITL-thrs': [0, 0, 0, 0, 0],
            'ASST-OTL-prop': [0, 0, 0, 0, 0],
            'ASST-OTL-thrs': [0, 0, 0, 0, 0],
            'B-ASST-ITL-prop': [0, 0, 0, 0, 0],
            'B-ASST-ITL-thrs': [0, 0, 0, 0, 0],
            'B-ASST-OTL-prop': [0, 0, 0, 0, 0],
            'B-ASST-OTL-thrs': [0, 0, 0, 0, 0],
        },
        index=signals.keys(),
    )

    entropies = pd.DataFrame(
        data={
            'CWT': [0, 0, 0, 0, 0],
            'SST': [0, 0, 0, 0, 0],
            'ASST-ITL-prop': [0, 0, 0, 0, 0],
            'ASST-ITL-thrs': [0, 0, 0, 0, 0],
            'ASST-OTL-prop': [0, 0, 0, 0, 0],
            'ASST-OTL-thrs': [0, 0, 0, 0, 0],
            'B-ASST-ITL-prop': [0, 0, 0, 0, 0],
            'B-ASST-ITL-thrs': [0, 0, 0, 0, 0],
            'B-ASST-OTL-prop': [0, 0, 0, 0, 0],
            'B-ASST-OTL-thrs': [0, 0, 0, 0, 0],
        },
        index=signals.keys(),
    )

    # Translate method string from printable to parameter:
    methd = {'thrs': 'threshold', 'prop': 'proportional'}

    sin_fig = plt.figure('Sine', dpi=300, figsize=(16/2.54, 6/2.54))
    sin_mse_fig = plt.figure('MSE - Sine', dpi=300)
    dqc_fig = plt.figure('Dual Quadratic Chirp', dpi=300, figsize=(16/2.54, 6/2.54))
    dqc_mse_fig = plt.figure('MSE - Dual Quadratic Chirp', dpi=300)
    lc_fig = plt.figure('Linear Chirp', dpi=300, figsize=(16/2.54, 6/2.54))
    lc_mse_fig = plt.figure('MSE - Linear Chirp', dpi=300)

    for signal_name, (f, signal) in signals.items():
        for itl, method in itertools.product([False, True], methd.keys()):
            itl_str = 'ITL' if itl else 'OTL'
            key = f'-{itl_str}-{method}'
            print(f'Analizing: {signal_name}{key}')

            figure_to_plot = None
            if_mse_figure_to_plot = None
            if signal_name == 'Dual Quadratic Chirp' and key == '-OTL-thrs':
                figure_to_plot = dqc_fig
                if_mse_figure_to_plot = dqc_mse_fig
            if signal_name == 'Sine' and key == '-OTL-prop':
                figure_to_plot = sin_fig
                if_mse_figure_to_plot = sin_mse_fig
            if signal_name == 'Linear Chirp' and key == '-ITL-prop':
                figure_to_plot = lc_fig
                if_mse_figure_to_plot = lc_mse_fig

            mse = get_mse_batched(
                signal,
                t,
                f,
                tr,
                method_fig=figure_to_plot,
                if_fig=if_mse_figure_to_plot,
                config=config,
                bLen=bLen,
                bMaxIters=bMaxIters,
                method=methd[method],
                threshold=0.5,
                itl=itl,
            )

            results.loc[signal_name, 'CWT'] = mse[0]
            results.loc[signal_name, 'SST'] = mse[1]
            results.loc[signal_name, 'ASST' + key] = mse[2]
            results.loc[signal_name, 'B-ASST' + key] = mse[3]
            entropies.loc[signal_name, 'CWT'] = mse[4][0]
            entropies.loc[signal_name, 'SST'] = mse[4][1]
            entropies.loc[signal_name, 'ASST' + key] = mse[4][2]
            entropies.loc[signal_name, 'B-ASST' + key] = mse[4][3]

    print("\nMSE Results:")
    print(results)

    print("\nEntropies:")
    print(entropies)

    # results.style.format('{:.4f}')
    results.style.format('{:.4f}').to_latex(
        str(parentDir / 'docs/latex/mse_table_signal.tex'),
        hrules=True,
        column_format='|r|c|c|c|c|c|c|c|c|c|c|',
    )

    # entropies.style.format('{:.4f}')
    entropies.style.format('{:.4f}').to_latex(
        str(parentDir / 'docs/latex/entropies_table.tex'),
        hrules=True,
        column_format='|r|c|c|c|c|c|c|c|c|c|c|',
    )

    ########################
    ## MSE vs Iterations: ##
    ########################
    iters = np.linspace(0, 4, 4, endpoint=False, dtype=int)

    f, signal = signals['Dual Quadratic Chirp']

    mseASSTIter = {
        'ITL/proportional': np.zeros_like(iters, dtype=float),
        'ITL/threshold': np.zeros_like(iters, dtype=float),
        'OTL/proportional': np.zeros_like(iters, dtype=float),
        'OTL/threshold': np.zeros_like(iters, dtype=float),
    }

    mseBASSTIter = {
        'ITL/proportional': np.zeros_like(iters, dtype=float),
        'ITL/threshold': np.zeros_like(iters, dtype=float),
        'OTL/proportional': np.zeros_like(iters, dtype=float),
        'OTL/threshold': np.zeros_like(iters, dtype=float),
    }

    for key, mseASST in mseASSTIter.items():
        for bMaxIter in iters:
            print(f'Analizing {key}, iter : {bMaxIter}')
            mse = get_mse_batched(
                signal,
                t,
                f,
                tr,
                config,
                bLen=bLen,
                bMaxIters=bMaxIter,
                method=key.split('/')[1],
                threshold=0.1,
                itl=(True if key.split('/', maxsplit=1)[0] == 'ITL' else False),
            )
            mseASST[bMaxIter] = mse[-3]
            mseBASSTIter[key][bMaxIter] = mse[-2]
        print(f'MSE: {mseASSTIter[key]}')
        print('----------')

    mseIterFig = plt.figure('MSE vs. Maximum Iterations', dpi=300)
    gs = mseIterFig.add_gridspec(1, 2)
    asstAx = plt.subplot(
        gs[0, 0],
    )
    asstAx.set_title('ASST')
    asstAx.set_xlabel('iterations', loc='right')
    asstAx.set_ylabel('MSE', loc='top')
    bAsstAx = plt.subplot(
        gs[0, 1],
    )
    bAsstAx.set_title('B-ASST')
    bAsstAx.set_xlabel('iterations', loc='right')
    bAsstAx.set_ylabel('MSE', loc='top')

    for key, mse in mseASSTIter.items():
        asstAx.plot(iters, mse, label=key)

    for key, mse in mseBASSTIter.items():
        bAsstAx.plot(iters, mse, label=key)

    asstAx.set_ylim(0, 1.5)
    bAsstAx.set_ylim(0, 1.5)

    asstAx.legend()
    bAsstAx.legend()
    mseIterFig.savefig(str(parentDir / 'docs/img/mse_vs_iters.pdf'))

    #################
    ## MSE vs SNR: ##
    #################

    f, signal = signals['Dual Quadratic Chirp']

    std_s = signal.std()
    snrsNumber = 30
    stds_n = np.linspace(0, 0.5, snrsNumber, endpoint=False, dtype=float) * std_s
    snrs = 20 * np.log10(1 / (stds_n) * std_s)

    print(f'Signal std = {std_s}')
    print(f'Noise stds = {stds_n}')
    print(f'SNRs = {snrs}')

    mean = 0
    num_samples = len(signal)

    bMaxIter = 1

    mseSnrASSTIter = {
        'ITL/proportional': np.zeros_like(stds_n, dtype=float),
        'ITL/threshold': np.zeros_like(stds_n, dtype=float),
        'OTL/proportional': np.zeros_like(stds_n, dtype=float),
        'OTL/threshold': np.zeros_like(stds_n, dtype=float),
    }

    mseSnrBASSTIter = {
        'ITL/proportional': np.zeros_like(stds_n, dtype=float),
        'ITL/threshold': np.zeros_like(stds_n, dtype=float),
        'OTL/proportional': np.zeros_like(stds_n, dtype=float),
        'OTL/threshold': np.zeros_like(stds_n, dtype=float),
    }

    mseSnrSST = np.zeros_like(stds_n, dtype=float)

    for key in mseASSTIter:
        for i, std_n in enumerate(stds_n):
            print(f'Analizing {key}, SNR : {snrs[i]}')
            noise = np.random.normal(mean, std_n, size=num_samples)
            mse = get_mse_batched(
                signal + noise,
                t,
                f,
                tr,
                config,
                bLen=bLen,
                bMaxIters=bMaxIter,
                method=key.split('/')[1],
                threshold=threshold,
                itl=(True if key.split('/')[0] == 'ITL' else False),
            )
            print(f'    MSE ---> {mse[-2]}')
            mseSnrASSTIter[key][i] = mse[-3]
            mseSnrBASSTIter[key][i] = mse[-2]
            mseSnrSST[i] = mse[1]
        print(f'MSE: {mseSnrASSTIter[key]}')
        print('----------')

    mseSnrFig = plt.figure('MSE vs. SNR', dpi=300)
    gs = mseSnrFig.add_gridspec(1, 2)
    asstSnrAx = plt.subplot(
        gs[0, 0],
    )
    asstSnrAx.set_title('ASST')
    asstSnrAx.set_xlabel('SNR [dB]', loc='right')
    asstSnrAx.set_ylabel('MSE', loc='top')
    bAsstSnrAx = plt.subplot(
        gs[0, 1],
    )
    bAsstSnrAx.set_title('B-ASST')
    bAsstSnrAx.set_xlabel('SNR [dB]', loc='right')
    bAsstSnrAx.set_ylabel('MSE', loc='top')

    for key, mse in mseSnrASSTIter.items():
        asstSnrAx.plot(snrs, mse, label=key)
    asstSnrAx.plot(snrs, mseSnrSST, ':', label='SST')

    for key, mse in mseSnrBASSTIter.items():
        bAsstSnrAx.plot(snrs, mse, label=key)
    bAsstSnrAx.plot(snrs, mseSnrSST, ':', label='SST')

    asstSnrAx.legend()
    bAsstSnrAx.legend()
    mseIterFig.savefig(str(parentDir / 'docs/img/mse_vs_snr.pdf'))

    plt.show()


if __name__ == "__main__":
    main()
