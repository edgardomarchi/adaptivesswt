#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:19:59 2021

@author: edgardo
"""
from dataclasses import dataclass

import numpy as np
import scipy.signal as sp
import soundfile as sf
from scipy.io import loadmat


@dataclass
class MeasurementData:
    radarI: np.ndarray
    radarQ: np.ndarray
    pcg: np.ndarray
    ecg2: np.ndarray
    ecg3: np.ndarray
    ecg: np.ndarray
    resp: np.ndarray
    fs: float = 1

def import_mat(filename: str):
    return loadmat(filename)


def raw2Data(dataRaw, normalize=True, filterLeads= True, removeLeadsOffset=True,
             removeIQoffset=True, balanceIQ=True):

    radarI = dataRaw['radar_I'].flatten()
    radarQ = dataRaw['radar_Q'].flatten()

    pcg = dataRaw['pcg_audio'].flatten()

    ecg2 = dataRaw['ecg_lead2'].flatten()
    ecg3 = dataRaw['ecg_lead3'].flatten()
    ecg = ecg2 - ecg3

    resp = dataRaw['respiration'].flatten()

    fs=dataRaw['Fs'][0][0]
    if filterLeads:
        #ECG:
        sosEcg =sp.iirfilter(5, 2*np.pi*15, btype='lowpass',
                             analog=False, ftype='butter',
                             output='sos', fs=2*np.pi*fs)
        ecg = sp.sosfilt(sosEcg, ecg)
        # Respiration:
        sosResp = sp.iirfilter(5, 2*np.pi*1.7, btype='lowpass',
                               analog=False, ftype='butter',
                               output='sos', fs=2*np.pi*fs)
        resp = sp.sosfilt(sosResp, resp)

    if removeLeadsOffset:
        pcg -= pcg.mean()
        ecg -= ecg.mean()
        resp -= resp.mean()

    if removeIQoffset:
        radarI -= radarI.mean()
        radarQ -= radarQ.mean()

    if balanceIQ:
        radarI *= (abs(radarQ).max() / abs(radarI).max())

    if normalize:
        max = np.array((abs(radarI).max(), abs(radarQ).max())).max()
        radarI /= max
        radarQ /= max
        pcg /= abs(pcg).max()
        ecg /= abs(ecg).max()
        resp /= abs(resp).max()

    return MeasurementData(radarI, radarQ, pcg, ecg2, ecg3, ecg, resp, fs)

def wav2Data(filename: str) -> MeasurementData:
    data, samplerate = sf.read(filename)
    return MeasurementData(data[:,0], data[:,1], np.zeros_like(data[:,0]), np.zeros_like(data[:,0]),
                           np.zeros_like(data[:,0]),np.zeros_like(data[:,0]),np.zeros_like(data[:,0]),
                           samplerate)

if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    import matplotlib.pyplot as plt
    import numpy as np

    plt.close('all')

    p6ap2L = '../../data/datasets_scidata_vsmdb/datasets/measurement_data_person6/PCG_front_radar_front/PCG_5L_radar_2L/apnea/DATASET_2017-02-06_09-27-58_Person 6.mat'
    p1ok4L = '../../data/datasets_scidata_vsmdb/datasets/measurement_data_person1/PCG_front_radar_front/radar_4L_PCG_2L/DATASET_2016-12-20_12-22-46_Person 1.mat'
    p2ok2L = '../../data/datasets_scidata_vsmdb/datasets/measurement_data_person2/PCG_front_radar_front/radar_2L_PCG_4L/DATASET_2017-02-27_10-04-44_Person 2.mat'
    p3ok4L = '../../data/datasets_scidata_vsmdb/datasets/measurement_data_person3/PCG_front_radar_front/radar_4L_PCG_2L/DATASET_2017-01-17_17-48-53_Person 3.mat'
    data = raw2Data(import_mat(p1ok4L))

    ts = 1/data.fs
    sig = data.radarI + 1j*data.radarQ

    stop = len(sig) * ts
    time = np.linspace(0, stop, len(sig))

    plt.figure('Datos importados')
    plt.plot(time, data.radarI, label='radar I')
    plt.plot(time, data.radarQ, label='radar Q')
    plt.plot(time, data.resp, label='Resp')
    plt.plot(time, data.pcg, label='PCG')
    plt.plot(time, data.ecg, label='ECG')
    plt.legend()
    plt.show()
