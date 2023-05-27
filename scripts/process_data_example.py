""" This script downloads and process data from the dataset below.

The script is intended to serve as an example for usage/configuration and evaluation of this package.

Test data used from:
Shi, Kilin; Schellenberger, Sven (2019): A dataset of radar-recorded heart sounds and vital signs
including synchronised reference sensor signals. figshare. Dataset.
https://doi.org/10.6084/m9.figshare.9691544.v1

Data URL: https://figshare.com/ndownloader/files/17357702

Usage:

$ python process_data_example.py -i dataset/datasets/measurement_data_person11/PCG_front_radar_front\
/PCG_2L_radar_4L/apnea/inhaled/DATASET_2017-02-16_10-56-05_Person\ 11.mat

"""

import argparse
import logging
import os
import urllib.request as rq
from os.path import abspath, dirname
from pathlib import Path
from zipfile import ZipFile

import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 10}

matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
import numpy as np

from adaptivesswt.configuration import Configuration
from adaptivesswt.sswt import reconstruct
from adaptivesswt.utils.freq_utils import calcScalesAndFreqs
from adaptivesswt.utils.import_utils import import_mat, raw2Data
from adaptivesswt.utils.process_data import analyze, extractPhase, intDecimate

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parentDir = Path(dirname(dirname(abspath(__file__))))

# Command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputFile", required=True, help="Input file to process")
args = vars(ap.parse_args())
dataFile = args['inputFile']

# Obtain dataset if not present # TODO: itegrate cmd args
datasetURL = 'https://figshare.com/ndownloader/files/17357702'
zipFileName = 'dataset.zip'
zipFileDir = './' + os.path.splitext(zipFileName)[0]

try:
    with ZipFile(zipFileName, 'r') as zipFile:
        if zipFile.testzip() is not None:
            raise FileNotFoundError('Zip File is damaged!')  # Bad
except FileNotFoundError:
    print('Downloading dataset...')
    rq.urlretrieve(datasetURL, zipFileName)
    print('Dataset downloaded')

if not os.path.isdir(zipFileDir):
    print('Extracting dataset...')
    with ZipFile(zipFileName, 'r') as zip_ref:
        zip_ref.extractall(zipFileDir)
    print('Done!')

print('Dataset Ready!')

# Data import
logger.info('Importing data...')
data = raw2Data(import_mat(dataFile), filterLeads=True)

signal, time = extractPhase(data)

# Get signals
((signalPCG, pcgFs), (signalPulse, pulseFs), (signalResp, respFs)) = intDecimate(
    signal, data.fs, 800, 40, 6
)

# Plotting parameters
dpi = 300

#### PCG
logger.info('Analizing PCG frequencies...')
configPCG = Configuration(
    min_freq=15,
    max_freq=200,
    num_freqs=50,
    ts=1 / pcgFs,
    wcf=1,
    wbw=8,
    wavelet_bounds=(-8, 8),
    threshold=abs(signalPCG).max() / 1e6,
    num_processes=4,
)

pcgIters = 2
pcgMethod = 'threshold'
pcgThreshold = abs(signalPCG).max() / 1e5
pcgItl = False

num_batchs = 6
batch_time = len(signalPCG) * pcgFs / num_batchs
pcgBLen = int(
    len(time[time <= (num_batchs * batch_time)][:: int(data.fs / pcgFs)]) // num_batchs
)
bPad = int(pcgBLen * 0.9)
configPCG.pad = bPad

plotPCG = True

sstPCG, aSstPCG, freqsPCG, pcgBatchs, pcgFig = analyze(
    signalPCG,
    configPCG,
    pcgIters,
    pcgMethod,
    pcgThreshold,
    pcgItl,
    pcgBLen,
    plot=plotPCG,
    tsst=True
)
if plotPCG:
    pcgFig.suptitle('PCG')  # type: ignore

_, sstFreqs, _, _ = calcScalesAndFreqs(
    respFs, configPCG.wcf, configPCG.min_freq, configPCG.max_freq, configPCG.num_freqs
)

signalPCGSynth = reconstruct(aSstPCG, configPCG.c_psi, freqsPCG)
sstSignalPCGSynth = reconstruct(sstPCG, configPCG.c_psi, sstFreqs)

signalPCGBSynthList = []
for (bAsstPCG, bFreqsPCG, _, _) in pcgBatchs:
    signalPCGBSynthList.append(reconstruct(bAsstPCG, configPCG.c_psi, bFreqsPCG))

signalPCGBSynth = np.array(signalPCGBSynthList[1:-1]).flatten()
signalPCGBSynth = np.concatenate(
    (signalPCGBSynthList[0], signalPCGBSynth, signalPCGBSynthList[-1])
)

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(time, -1 * data.ecg / abs(data.ecg).max(), label='ECG')
ax[0].plot(time, -1 * data.pcg / abs(data.pcg).max(), label='PCG')
ax[1].plot(
    time[:: int(data.fs / pcgFs)],
    sstSignalPCGSynth / abs(sstSignalPCGSynth)[200:-200].max(),
    label=f'SSWT - Radar',
)
ax[2].plot(
    time[:: int(data.fs / pcgFs)],
    signalPCGSynth / abs(signalPCGSynth)[200:-200].max(),
    'g',
    label=f'SSWT ADPT - Radar',
)
ax[3].plot(
    time[:: int(data.fs / pcgFs)],
    signalPCGBSynth / abs(signalPCGBSynth[200:-200]).max(),
    'r',
    label=f'SSWT B-ADPT - Radar',
)
for axis in ax:
    axis.legend()
fig.suptitle('PCG')


#### Pulse
logger.info('Analizing Pulse frequencies...')
configPulse = Configuration(
    min_freq=1,
    max_freq=3,
    num_freqs=16,
    ts=1 / pulseFs,
    wcf=1,
    wbw=10,
    wavelet_bounds=(-8, 8),
    threshold=abs(signalPulse).max() / 1e5,
    num_processes=4,
)

pulseIters = 1
pulseMethod = 'proportional'
pulseThreshold = abs(signalPulse).max() / 10000
pulseItl = True

num_batchs = 6
batch_time = len(signalPulse) * pulseFs / num_batchs
pulseBLen = (
    int(
        len(time[time <= (num_batchs * batch_time)][:: int(data.fs / pulseFs)])
        // num_batchs
    )
    + 1
)
bPad = int(pulseBLen * 0.9)
configPulse.pad = bPad

plotPulse = True

sstPulse, aSstPulse, freqsPulse, pulseBatchs, pulseFig = analyze(
    signalPulse,
    configPulse,
    pulseIters,
    pulseMethod,
    pulseThreshold,
    pulseItl,
    pulseBLen,
    plotPulse,
)
if plotPulse:
    #pulseFig.suptitle('Pulse')  # type: ignore
    pulseFig.savefig(  # type: ignore
        parentDir / 'docs/img/pulse_method_comparison.png',
        dpi=dpi,
        bbox_inches='tight',
    )

signalPulseSynth = reconstruct(aSstPulse, configPulse.c_psi, freqsPulse)

signalPulseBSynthList = []
for (bAsstPulse, bFreqsPulse, _, _) in pulseBatchs:
    signalPulseBSynthList.append(
        reconstruct(bAsstPulse, configPulse.c_psi, bFreqsPulse)
    )

signalPulseBSynth = np.array(signalPulseBSynthList[1:-1]).flatten()
signalPulseBSynth = np.concatenate(
    (signalPulseBSynthList[0], signalPulseBSynth, signalPulseBSynthList[-1])
)

fig, ax = plt.subplots(1, 1, dpi=dpi)
ax.plot(time, -1 * data.ecg / abs(data.ecg).max(), label='ECG')
ax.plot(
    time[:: int(data.fs / pulseFs)],
    signalPulseSynth / abs(signalPulseSynth).max(),
    label=f'SSWT ADPT - Radar',
)
ax.plot(
    time[:: int(data.fs / pulseFs)],
    signalPulseBSynth / abs(signalPulseBSynth).max(),
    label=f'SSWT B-ADPT - Radar',
)
ax.legend()
fig.suptitle('Pulse')

#### Respiration:
logger.info('Analizing Respiration frequencies...')
configResp = Configuration(
    min_freq=0.2 * respFs / len(signalResp),
    max_freq=1,
    num_freqs=16,
    ts=1 / respFs,
    wcf=1,
    wbw=10,
    wavelet_bounds=(-8, 8),
    threshold=abs(signalResp).max() / 10000,
    num_processes=4,
)
print(f'Min freq: {configResp.min_freq}')

respIters = 1
respMethod = 'proportional'
respThreshold = abs(signalPCG).max() / 10000
respItl = True

num_batchs = 6
batch_time = len(signalResp) * respFs / num_batchs
respBLen = int(
    len(time[time <= (num_batchs * batch_time)][:: int(data.fs / respFs)]) // num_batchs
)
bPad = int(respBLen * 0.9)
configResp.pad = bPad

plotResp = False

sstResp, aSstResp, freqsResp, respBatchs, respFig = analyze(
    signalResp,
    configResp,
    respIters,
    respMethod,
    respThreshold,
    respItl,
    respBLen,
    plotResp,
)
if plotResp:
    respFig.suptitle('Respiration')  # type: ignore

_, sstFreqs, _, _ = calcScalesAndFreqs(
    respFs, configResp.wcf, configResp.min_freq, configResp.max_freq, configResp.num_freqs
)

signalRespSynth = reconstruct(aSstResp, configResp.c_psi, freqsResp)
sstSignalRespSynth = reconstruct(sstResp, configResp.c_psi, sstFreqs)

signalRespBSynthList = []
for (bAsstResp, bFreqsResp, _, _) in respBatchs:
    signalRespBSynthList.append(reconstruct(bAsstResp, configResp.c_psi, bFreqsResp))

signalRespBSynth = np.array(signalRespBSynthList[1:-1]).flatten()
signalRespBSynth = np.concatenate(
    (signalRespBSynthList[0], signalRespBSynth, signalRespBSynthList[-1])
)

fig, ax = plt.subplots(1, 1, dpi=dpi)
ax.plot(
    time, -1 * data.resp / data.resp.max(), label='Respiration (thermal)'
)  # Gets inverted due thermal method
ax.plot(time[:: int(data.fs / respFs)], sstSignalRespSynth, label=f'SSWT- Radar')
ax.plot(time[:: int(data.fs / respFs)], signalRespSynth, label=f'SSWT ADPT - Radar')
ax.plot(time[:: int(data.fs / respFs)], signalRespBSynth, label=f'SSWT B-ADPT - Radar')
ax.legend()
fig.suptitle('Respiration')

plt.show()
