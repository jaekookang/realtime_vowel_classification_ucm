'''
This script extracts acoustic features and the corresponding labels 
from XRMB dataset.

Options:
- [] Vowel only
- [] Vowel + Consonants
- [] All frames (including silence)

2018-08-21
'''

import ipdb as pdb
import os
import sys
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
from speech_features import SpeechFeatures


class hparams:
    # Parameters
    num_freq = 1025
    sample_rate = 21739
    win_size = 0.025
    win_step = 0.01
    preemp = 0.97
    nfilt = 40  # filter no.


# Get parameters
hp = hparams()


def read_mat(matfile):
    '''Read matfile'''
    FileID = re.sub('.mat', '', os.path.basename(matfile))
    return loadmat(matfile)[FileID]


def find_elements(pattern, my_list):
    '''Find elements in a list'''
    elements = []
    index = []

    for i, l in enumerate(my_list):
        if re.search(pattern, l):
            elements.append(my_list[i])
            index.append(i)
    return index, elements


if __name__ == '__main__':
    '''
    여기서부터 다시 할것.
    무엇을 변환하고 정제할지 계획을 세우고 코드를 짤 것!
    '''
    pass
