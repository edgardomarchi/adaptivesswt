#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def renyi_entropy(tfMatrix:np.ndarray, alpha:int = 3) -> float:
    absMatrSq = np.abs(tfMatrix)**2
    E = absMatrSq.sum()
    if alpha > 1:
        entropy = 1/(1-alpha) * np.log2(((absMatrSq / E )**alpha).sum())
    else:
        entropy = np.log2((absMatrSq / E ).sum())
    return entropy
