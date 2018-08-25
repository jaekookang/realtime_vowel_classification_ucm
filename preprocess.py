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

import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
import textgrid
from tqdm import tqdm
from speech_features import SpeechFeatures
from sklearn.decomposition import PCA


class hparams:
    # Parameters
    phones = ['AE1', 'AH1', 'AO1', 'EH1', 'IH1', 'AA1', 'IY1', 'UW1', 'UH1']
    num_freq = 1025
    sample_rate = 21739
    win_size = 0.025
    win_step = 0.01
    preemp = 0.97
    n_comp = 40  # PCs
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


def read_textgrid(tg_file):
    '''Read textgrid and organize'''
    T = textgrid.TextGrid()
    T.read(tg_file)
    tier = T.getFirst('phone').intervals
    labs = []
    times = np.zeros((len(tier), 2))  # (min, max)
    for i in range(len(tier)):
        labs.append(tier[i].mark)
        times[i, :] = [tier[i].minTime, tier[i].maxTime]
    return labs, times


def extract_features(wav_file, tg_file, segment, time='center'):
    '''Extract FFTs'''
    # Initialize SpeechFeatures
    S = SpeechFeatures(wav_file, hp.win_size, hp.win_step, hp.num_freq,
                       hp.nfilt, None, win_fun=np.hamming, pre_emp=hp.preemp)
    # Read textgrid
    labs, times = read_textgrid(tg_file)
    samples = (times*S.srate).astype(np.int32)
    # Iterate over the provided segments
    # fft_all = np.array([], dtype=np.float32).reshape(0, hp.num_freq//2+1)
    fft_all = np.array([], dtype=np.float32).reshape(0, hp.nfilt)
    idx, _ = find_elements(segment, labs)
    if len(idx) > 0:
        for i in idx:
            # Extract sample
            begT, endT = samples[i, 0], samples[i, 1]
            sig_part = S.sig[begT:endT]
            # Get log spectorgram
            # _, _, logspec = S.get_fft(sig=sig_part)
            # Apply linear filters
            fftlin, _ = S.get_fft(sig=sig_part, linfilt=True)
            # Slice sample
            if time == 'center':
                # fft = logspec[logspec.shape[0]//2, :]
                fft = fftlin[fftlin.shape[0]//2, :]
            elif time == 'all':
                # fft = logspec
                fft = fftlin
            else:
                raise Exception(f'time={time} is not supported yet')
            # Save
            fft_all = np.vstack([fft_all, fft])
        return fft_all
    else:
        return None


def get_pca(data, n_comp):
    '''Reduce the dimension of the data into n_comp
    (n x p) --> (n x n_comp)
    '''
    pca = PCA(n_components=n_comp)
    return pca, pca.fit_transform(data)


if __name__ == '__main__':
    # Get directories
    if sys.platform == 'darwin':
        XRMB_DIR = '/Volumes/Transcend/_DataArchive/WisconsinArchives/XRMB_WAV_TEXTGRID'
    elif sys.platform == 'linux':
        XRMB_DIR = '/home/zzandore/zzandore/dataset/XRMB_WAV_TEXTGRID'
    else:
        raise Exception('OS should be either Mac or Linux')

    # Get file paths
    wavs = sorted(glob.glob(os.path.join(XRMB_DIR, '**', '**', '*.wav')))
    tgs = sorted(glob.glob(os.path.join(XRMB_DIR, '**', '**', '*.TextGrid')))
    # Check file names
    for w, t in zip(wavs, tgs):
        w_id = os.path.split(w)[1].split('.')[0]
        t_id = os.path.split(t)[1].split('.')[0]
        assert w_id == t_id

    # fft_all = np.array([], dtype=np.float32).reshape(0, hp.num_freq//2+1)
    fft_all = np.array([], dtype=np.float32).reshape(0, hp.nfilt)
    labels = []
    for i, (wav, tg) in tqdm(enumerate(zip(wavs, tgs)), total=len(wavs)):
        # e.g. JW11_TP003
        spkr_id = os.path.split(wav)[1].split('.')[0]

        for phn in hp.phones:
            fft = extract_features(wav, tg, phn, time='center')
            if fft is not None:
                # Save
                fft_all = np.vstack([fft_all, fft])
                for _ in range(fft.shape[0]):
                    labels.append(phn)

    # Get PCA
    pca, fftpca = get_pca(fft_all, hp.n_comp)
    # Make phone2idx, idx2phone
    phn2idx = {phn: i for i, phn in enumerate(hp.phones)}
    idx2phn = {phn2idx[phn]: phn for phn in hp.phones}

    # Save
    np.save('fft_all.npy', fft_all)
    np.save('fft_linfilt.npy', fft_all)
    np.save('fft_linfilt_pca.npy', pca)
    np.save('fft_linfilt_pca_reduced.npy', fftpca)

    np.save('labels.npy', labels)
    np.save('phn2idx.npy', phn2idx)
    np.save('idx2phn.npy', idx2phn)

    print('Finished')
