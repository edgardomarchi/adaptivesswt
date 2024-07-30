import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Logging configuration
logging.basicConfig(filename='sswt_test.log', filemode='w',
                    format='%(levelname)s - %(asctime)s - %(name)s:\n %(message)s')

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('adaptivesswt').setLevel(logging.INFO)

logger = logging.getLogger('')
logger.setLevel(logging.INFO)


from adaptivesswt import getBackend, sswt
from adaptivesswt.configuration import Configuration
from adaptivesswt.sswt import calcScalesAndFreqs, reconstruct, reconstructCWT
from adaptivesswt.utils import signal_utils as generator
from adaptivesswt.utils.measures_utils import renyi_entropy

# Plot configuration
font = {'family': 'normal', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)
plt.rcParams['text.usetex'] = True


###########################
# Test signal generation: #
###########################
stopTime = 12
fs = 400
signalLen = stopTime * fs

t, step = np.linspace(0, stopTime, signalLen, endpoint=False, retstep=True)
ts = float(step)  # type: ignore # np.Floating[Any] to float

# signal = generator.testSine(t, 0.2) + generator.testSine(t,1) + generator.testSine(t, 5) + generator.testSine(t,10)
# f, signal = generator.testSig(t)
# f, signal = generator.crossChrips(t, 2, 8, 2)
# f, signal = generator.testChirp(t, 3, 6)
# _, signal = generator.quadraticChirp(t, 1, 30)
f, signal = generator.dualQuadraticChirps(t, (8,5),(2,4))
# signal = np.zeros_like(t)
# signal[fs:2*fs]=1.0

##########################
# Analysis configuration #
##########################
config = Configuration(
    min_freq=0.1,
    max_freq=11,
    num_freqs=64,
    ts=ts,
    wcf=1,
    wbw=4,
    wavelet_bounds=(-8,8),
    threshold=signal.max()/(100),
    pad=256,
    plot_filters=False)


#####################
# Method comparison #
#####################
scales, _, _, _ = calcScalesAndFreqs(ts, config.wcf, config.min_freq, config.max_freq, config.num_freqs)

sst, cwt, freqs, wab, tail = sswt(signal, **config.asdict())
rentrCWT = renyi_entropy(cwt,3)
rentrSST = renyi_entropy(sst,3)
print(f'Rènyi entropy of CWT = {rentrCWT:.4f}')
print(f'Rènyi entropy of SST = {rentrSST:.4f}')
print('--------------------------------------------------\n')

## Plotting ##
mainFig = plt.figure('Comparación de métodos')
gs = mainFig.add_gridspec(1, 3)
mainAxes  = gs.subplots(sharex='col', sharey='row')
mainFig.set_layout_engine('tight')

mainAxes[1].pcolormesh(t, freqs, np.abs(cwt), cmap='viridis', shading='gouraud')
mainAxes[1].set_title('Wavelet Transform')
mainAxes[1].set_xlabel('t [s]', loc='right')
mainAxes[1].set_ylabel('f [Hz]', loc='top')
mainAxes[2].pcolormesh(t, freqs, np.abs(sst), cmap='viridis', shading='gouraud')
mainAxes[2].set_title('Synchrosqueezing Transform')
mainAxes[2].set_xlabel('t [s]', loc='right')
mainAxes[2].set_ylabel('f [Hz]', loc='top')

signalR_cwt = reconstructCWT(cwt, config.wav, scales, freqs)
signalR_cwt /= signalR_cwt.max()
signalR_sst = reconstruct(sst, config.c_psi, freqs)

mainAxes[0].plot(t, f[0], label='Comp. 0')
mainAxes[0].plot(t, f[1], label='Comp. 1')
mainAxes[0].set_xlabel('t [s]', loc='right')
mainAxes[0].set_ylabel('f [Hz]', loc='top')
mainAxes[0].legend()
mainAxes[0].set_title('Frecuencias intstantáneas')

#############################
# Transform size comparison #
#############################
config.num_freqs = 16
sst16, _, freqs16, _, _ = sswt(signal, **config.asdict())
config.num_freqs = 32
sst32, _, freqs32, _, _ = sswt(signal, **config.asdict())
config.num_freqs = 64
sst64, _, freqs64, _, _ = sswt(signal, **config.asdict())
config.num_freqs = 128
sst128, _, freqs128, _, _ = sswt(signal, **config.asdict())

## Plotting ##
sizeFig = plt.figure('Comparación de K')
gsSize = sizeFig.add_gridspec(2, 2)
sizeAxes  = gsSize.subplots(sharex='col', sharey='row')
sizeFig.set_layout_engine('tight')

sizeAxes[0, 0].pcolormesh(t, freqs16, np.abs(sst16), cmap='viridis', shading='gouraud')
sizeAxes[0, 0].set_title('K = 16')
sizeAxes[0, 0].set_xlabel('t [s]', loc='right')
sizeAxes[0, 0].set_ylabel('f [Hz]', loc='top')

sizeAxes[0, 1].pcolormesh(t, freqs32, np.abs(sst32), cmap='viridis', shading='gouraud')
sizeAxes[0, 1].set_title('K = 32')
sizeAxes[0, 1].set_xlabel('t [s]', loc='right')
sizeAxes[0, 1].set_ylabel('f [Hz]', loc='top')

sizeAxes[1, 0].pcolormesh(t, freqs64, np.abs(sst64), cmap='viridis', shading='gouraud')
sizeAxes[1, 0].set_title('K = 64')
sizeAxes[1, 0].set_xlabel('t [s]', loc='right')
sizeAxes[1, 0].set_ylabel('f [Hz]', loc='top')

sizeAxes[1, 1].pcolormesh(t, freqs128, np.abs(sst128), cmap='viridis', shading='gouraud')
sizeAxes[1, 1].set_title('K = 128')
sizeAxes[1, 1].set_xlabel('t [s]', loc='right')
sizeAxes[1, 1].set_ylabel('f [Hz]', loc='top')

rentrSST16 = renyi_entropy(sst16,3)
rentrSST32 = renyi_entropy(sst32,3)
rentrSST64 = renyi_entropy(sst64,3)
rentrSST128 = renyi_entropy(sst128,3)

print(f'Rènyi entropy of SST with K=16 : {rentrSST16:.4f}')
print(f'Rènyi entropy of SST with K=32 : {rentrSST32:.4f}')
print(f'Rènyi entropy of SST with K=64 : {rentrSST64:.4f}')
print(f'Rènyi entropy of SST with K=128 : {rentrSST128:.4f}')
print('--------------------------------------------------\n')

###############################
#### Transforms comparison ####
###############################
tsstFig, tsstAx = plt.subplots(1,5)
tsstFig.set_layout_engine('tight')

sst, cwt, freqs, tab, tail = sswt(signal, **config.asdict())
tsstAx[0].pcolormesh(t, freqs, np.abs(cwt[:-1,:-1]), cmap='viridis', shading='flat') #'gouraud')
tsstAx[0].set_title('Wavelet Transform')
tsstAx[1].pcolormesh(t, freqs, np.abs(sst[:-1,:-1]), cmap='viridis', shading='flat') #'gouraud')
tsstAx[1].set_title('Synchrosqueezing Transform')

config.transform = 'tsst'
tsst, cwt, freqs, tab, tail = sswt(signal, **config.asdict())
tsstAx[2].pcolormesh(t, freqs, np.abs(tsst[:-1,:-1]), cmap='viridis', shading='flat') #'gouraud')
tsstAx[2].set_title('Time synchrosqueezing Transform')

config.transform = 'tfr'
tfr, cwt, freqs, tab, tail = sswt(signal, **config.asdict())
tsstAx[3].pcolormesh(t, freqs, np.abs(tfr)[:-1,:-1], cmap='viridis', shading='flat') #'gouraud')
tsstAx[3].set_title('Time Frequency Reasignment')

config.transform = 'set'
tfr, cwt, freqs, tab, tail = sswt(signal, **config.asdict())
tsstAx[4].pcolormesh(t, freqs, np.abs(tfr)[:-1,:-1], cmap='viridis', shading='flat') #'gouraud')
tsstAx[4].set_title('Synchro-Extracting Transform')

print(f'Max values: CWT={abs(cwt).max()}, SST={abs(sst).max()}, TSST={abs(tsst).max()}, TFR={abs(tfr).max()}\n')


################
#### Timing ####
################
import timeit

passes = 10
config.pad = 0
time = timeit.timeit("lambda: sswt(signal, **config.asdict())", globals=globals(), number=passes)
print(f'Excecution time for {passes} passes and {getBackend()} = {time:.4E}s')
print(f'Execution time per signal second (<1: Real-time, >1: Not real-time) = {(time / stopTime / passes):.4E} s/s')

plt.show()
