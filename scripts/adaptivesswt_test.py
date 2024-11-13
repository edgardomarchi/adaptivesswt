import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp

from adaptivesswt import adaptive_sswt, adaptive_sswt_slidingWindow, sswt
from adaptivesswt.configuration import Configuration
from adaptivesswt.sswt import reconstruct, reconstructCWT
from adaptivesswt.utils import signal_utils as generator
from adaptivesswt.utils.freq_utils import getScale
from adaptivesswt.utils.measures_utils import renyi_entropy
from adaptivesswt.utils.plot_utils import plot_batched_tf_repr, plot_tf_repr

# Logging configuration
logging.basicConfig(filename='adaptivesswt_test.log', filemode='w',
                    format='%(levelname)s - %(asctime)s - %(name)s:\n %(message)s')

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('adaptivesswt').setLevel(logging.INFO)

logger = logging.getLogger('')
logger.setLevel(logging.INFO)

# Plot configuration
font = {'family': 'normal', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)
plt.rcParams['text.usetex'] = True

 #%% Simulation Parameters

stopTime = 12
fs = 200
signalLen = int(stopTime * fs)

wcf = 1
wbw = 10

max_freq = 50
min_freq = 10
num_freqs = 16
specgram_num_freqs = int(num_freqs / (max_freq - min_freq) * (fs/2))

t, step = np.linspace(0, stopTime, signalLen, endpoint=False, retstep=True)
ts = float(step)

#%% Setup Figures for Different Transforms comparison

compFig = plt.figure()
gs = compFig.add_gridspec(2, 3)

ifAx = plt.subplot(
        gs[0, 0],
    )
wtAx = plt.subplot(
        gs[0, 2],
    )
wtAx.sharey(ifAx)
spAx = plt.subplot(
        gs[0, 1],
    )
spAx.sharey(ifAx)
sstAx = plt.subplot(
        gs[1, 0],
    )
sstAx.sharex(ifAx)
asstAx = plt.subplot(
        gs[1, 1],
    )
asstAx.sharey(sstAx)
bAsstAx = plt.subplot(
        gs[1, 2],
    )
bAsstAx.sharey(sstAx)

gs.tight_layout(compFig) #, rect=[0, 0.03, 1, 0.95])

#%% Test signals

# f, sig = generator.testChirp(t, 10, 25)
# _, sig = testUpDownChirp(t,1,10)
# f, sig = generator.quadraticChirp(t, 40, 30)
frqs, sig = generator.dualQuadraticChirps(t, (28, 30), (42, 38))
# f, sig = generator.testSig(t)
# frqs, sig = generator.tritone(t, 20, 10 * np.pi, 40)
# f, sig = generator.testSine(t,15)
# sig = generator.delta(t, 2)
# frqs, sig = generator.crossChrips(t, 20, 40, 3)

split_freq = 34

f1, f2 = frqs
ifAx.plot(t[: len(sig)], f1)
ifAx.plot(t[: len(sig)], f2)
# ifAx.plot(t[: len(sig)], f3)
ifAx.set_title('Instantaneous frequency')
ifAx.set_xlabel('t [s]', loc='right')
ifAx.set_ylabel('f [Hz]',loc='top')

#%% SST, CWT and Spectrogram

sstConfig = Configuration(
    min_freq=min_freq,
    max_freq=max_freq,
    num_freqs=num_freqs,
    ts=ts,
    wcf=wcf,
    wbw=wbw,
    wavelet_bounds=(-8, 8),
    threshold=sig.max() / (100)
    )

config = Configuration(
    min_freq=min_freq,
    max_freq=max_freq,
    num_freqs=num_freqs,
    ts=ts,
    wcf=wcf,
    wbw=wbw,
    wavelet_bounds=(-8, 8),
    threshold=sig.max() / (100)
    )

sp_f, sp_t, ssp = sp.spectrogram(sig, fs=fs, nperseg=specgram_num_freqs, noverlap=specgram_num_freqs-1)
valid_fqs = np.logical_and(sp_f >= min_freq, sp_f <= max_freq)

plot_tf_repr(ssp[valid_fqs,:], sp_t, sp_f[valid_fqs], spAx)
spAx.set_title('Spectrogram')

sst, cwt, freqs, wab, tail = sswt(sig, **sstConfig.asdict())
reCwt = renyi_entropy(cwt)

print(f'Entropy of CWT = {reCwt}')
reSst = renyi_entropy(sst)
print(f'Entropy of SST = {reSst}')

plot_tf_repr(cwt, t, freqs, wtAx)
wtAx.set_title('Wavelet Transform')

plot_tf_repr(sst, t, freqs, sstAx)
sstAx.set_title('Synchrosqueezing Transform')

scales = getScale(freqs, config.ts, config.wcf)
signalR_cwt = reconstructCWT(cwt, config.wav, scales, freqs)  # type: ignore # scales is always a ndarray
signalR_sst = reconstruct(sst, config.c_psi, freqs)

# TODO: Define a frontier and keep one frequency for plotting
f_sst = freqs[np.argmax(abs(sst), axis=0)]
f_cwt = freqs[np.argmax(abs(cwt), axis=0)]

#%% Adaptive SST and minibatch A SST

maxIters = 2
threshold = config.threshold * 2
method = 'proportional'
itl = False

batch_time = 2
num_batchs = int(stopTime // batch_time)

asst, aFreqs, wab, tail = adaptive_sswt(
        sig, maxIters, method, threshold, itl, **config.asdict()
    )
reAsst = renyi_entropy(asst)
print(f'Entropy of ASST = {reAsst}')

f_asst = aFreqs[np.argmax(abs(asst), axis=0)]

plot_tf_repr(asst, t, aFreqs, asstAx)
asstAx.set_title('Adaptive SST')

signalR_asst = reconstruct(asst, config.c_psi, aFreqs)

## Minibatch
bLen = int(len(t[t <= (num_batchs * batch_time)]) // num_batchs)
bPad = int(bLen * 0.9)
config.pad = bPad
bMaxIters = 5

batchs = adaptive_sswt_slidingWindow(
    bLen, sig, bMaxIters, method, threshold, itl, **config.asdict()
    )
plot_batched_tf_repr(batchs, ts, bAsstAx)

compFig.tight_layout()

fBatchList = []
# Recover instantaneous frequencies:
for (asswt, freqs, wab, tail) in batchs:
    fBatchList.append(freqs[np.argmax(abs(asswt), axis=0)])

f_batch = np.array(fBatchList[1:-1]).flatten()
f_batch = np.concatenate((fBatchList[0], f_batch, fBatchList[-1]))

#%% Instantaneous frequencies across transforms

fig, ax = plt.subplots(1)
ax.plot(t[: len(sig)], f_cwt, label='CWT')
ax.plot(t[: len(sig)], f_sst, label='SST')
ax.plot(t[: len(sig)], f_asst, label='ASST')
ax.plot(t[: len(f_batch)], f_batch, label=f'ASST ({batch_time}s batchs)')
ax.plot(t[: len(sig)], f1, label='Signal')
ax.legend()
fig.suptitle('Instantaneous frequencies')

fig, ax = plt.subplots(1)
ax.plot(
    t[: len(signalR_sst)],
        sig[: len(signalR_sst)],
        label='Original Signal',
        alpha=0.8,
    )
ax.plot(t[: len(signalR_sst)], signalR_sst, label='SST Signal')
ax.plot(t[: len(signalR_asst)], signalR_asst, label='ASST Signal')
ax.legend()
fig.suptitle('Signal reconstruction')

mse_cwt = (f1 - f_cwt) ** 2
mse_sst = (f1 - f_sst) ** 2
mse_asst = (f1 - f_asst) ** 2
mse_asst_batch = (f1[: len(f_batch)] - f_batch) ** 2

mse_cwt_total = mse_cwt.sum() / len(mse_cwt)
mse_sst_total = mse_sst.sum() / len(mse_sst)
mse_asst_total = mse_asst.sum() / len(mse_asst)
mse_asst_batch_total = mse_asst_batch.sum() / len(mse_asst_batch)

fig, ax = plt.subplots(1)
ax.plot(t[: len(mse_cwt)], mse_cwt, label=f'CWT - MSE = {mse_cwt_total:.3}')
ax.plot(t[: len(mse_sst)], mse_sst, label=f'SST - MSE = {mse_sst_total:.3}')
ax.plot(t[: len(mse_asst)], mse_asst, label=f'ADPT SST - MSE = {mse_asst_total:.3}')
ax.plot(
    t[: len(f_batch)],
    mse_asst_batch,
    label=f'B-ASST - MSE = {mse_asst_batch_total:.3}',
    )
ax.legend()
fig.suptitle('MSE(f)')

#%% Plot bands
print("Press 'WB' to plot wavelet bands:")
if input() == 'WB':
    maxIters = 2
    threshold = 0.15  # config.threshold * 3
    method = 'threshold'
    itl = False
    config.plot_filters = True

    asst, aFreqs, wab, tail = adaptive_sswt(
        sig, maxIters, method, threshold, itl, **config.asdict()
        )

#%% Compare methods

K=24

fs = 256
stopTime = 16

test_config = Configuration(
    min_freq = 1,
    max_freq = fs/2,
    num_freqs= K,
    ts=1/fs,
    wcf=1,
    wbw=14,
    wavelet_bounds=(-8,8),
    transform='sst'
    )

max_iters = 1
method = 'proportional'
itl = True

t, step = np.linspace(0, stopTime, int(fs*stopTime), endpoint=False, retstep=True)
ts = float(step)

# f, signal = generator.matlab_comparison_signals(t, ts)
f, signal = generator.dualQuadraticChirps(t, (28, 30), (42, 38))

import time as clock

prev = clock.time()
asst, aFreqs, _, _ = adaptive_sswt(
    signal, max_iters, method, threshold, itl, **test_config.asdict()
    )

logger.info('Time to run ASST with K = %s: %s s', K, clock.time()-prev)
mCompFig, mCompAx = plt.subplots(1)
# mCompFig.suptitle('Signal for comparison')
plot_tf_repr(asst, t, aFreqs, mCompAx)
mCompAx.invert_yaxis()

plt.show(block=False)

resp = input('Press "T" to time performance. Press any ohter key to end...')
plt.close('all')

if resp == 'T':
    import timeit

    bSswt_fix = lambda: adaptive_sswt_slidingWindow(
        bLen, sig, bMaxIters, method, threshold, itl, **config.asdict()
        )
    timingProc = timeit.timeit(bSswt_fix, number=5) / 5
    timingProcPS = timingProc / len(sig)
    print(f'Average execution time: {timingProc}')
    print(f'Number of processes : {config.num_processes}')
    print(f'Average timing per signal sample = {timingProcPS} s/s')
