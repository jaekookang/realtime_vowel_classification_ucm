'''
Realtime player for frame-wise phone classifier
based on XRMB dataset

2018-08-23
'''

import ipdb as pdb
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import os
import re
import sys
import numpy as np
import sounddevice as sd
from speech_features import SpeechFeatures

import tensorflow as tf
from train import *
from numpy_ringbuffer import RingBuffer
import threading
import queue


def update_plot(frame):
    '''Update plots'''
    global data, obj, txt, buf, q
    while True:
        try:
            _data, _ = q.get_nowait()
        except queue.Empty:
            break
        buf.extend(_data.flatten())  # append at the end
        # Update data
        S._pre_emp(sig=buf)
        filtered, _ = S.get_fft(linfilt=True)
        data['linfilt'] = filtered  # (1,40)
        X_reduced = pca.transform(filtered)  # (1,15)
        X_recovered = pca.inverse_transform(X_reduced)  # (1,40)
        X_std = X_scaler.transform(X_reduced)  # (1,15)
        prob = forward(X_std, m_params)
        # Update plots
        obj['linfilt'][0].set_ydata(X_recovered)
        for i, prb in enumerate(prob[0]):
            obj['bars'][i].set_height(prb)
            if np.max(prob) == prb:
                # Mark the highest probable vowel
                obj['bars'][i].set_color('r')
                txt.set_text(f'{phones[i]}: {prb:.1f} %')
                txt.set_position((i, prb))
                txt.set_size(12)
                txt.set_color('black')
            else:
                obj['bars'][i].set_color('b')
                txt.set_text('')
            '''
            왜 텍스트가 제대로 안뜨는지 확인할것
            '''
    return [obj['linfilt'][0]] + [b for b in obj['bars']] + [txt]


def init_plot(dada, obj):
    # global p
    '''Initialize axes and plot obj/data'''
    # Set plot data
    data['linfilt'] = np.zeros((p.nfilt, 1))  # (40,1)
    data['bars'] = np.zeros(len(phones))  # (9,)
    # Make figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Make plots
    obj['linfilt'] = ax1.plot(data['linfilt'], color='black',
                              marker='.', linestyle='dashed',
                              markersize=10)
    obj['bars'] = ax2.bar(range(len(phones)), data['bars'])
    # Set axis & axes
    ax1.set_title('Linear-filtered FFT')
    ax1.set_xticks(range(p.nfilt))
    ax1.set_xticklabels([f'{t+1:d}' for t in range(p.nfilt)])
    ax1.set_ylim([60, 220])
    ax1.tick_params(rotation=90)
    ax1.set_xlabel('40 Filters (reconstructed from 15 PCs)')

    ax2.set_title('Vowel probabilities')
    ax2.set_xticks(range(len(phones)))
    ax2.set_xticklabels([p for p in phones])
    ax2.yaxis.set_ticks(np.linspace(0, 1, 11))
    ax2.set_yticklabels([f'{t:.1f}' for t in np.linspace(0, 1, 11)])
    ax2.set_xlabel('9 selected vowels')
    ax2.set_ylabel('Probability')
    # Set label holder
    txt = ax2.text(len(phones), 0.9, 'TEST')
    fig.tight_layout(pad=3)
    return fig, data, obj, txt


def init_buf():
    '''Initialize buffer'''
    buf = RingBuffer(p.winsamp, dtype=np.int16)
    eps = np.finfo(float).eps
    for _ in range(p.winsamp):
        buf.append(eps)
    return buf


class Record(threading.Thread):

    def __init__(self, i_que):
        '''Start audio input streaming'''
        threading.Thread.__init__(self)
        self.i_que = i_que
        self.kill_received = False

        # Open inputstream
        self.streamflow = sd.InputStream(
            device=p.device, channels=p.channels, samplerate=p.samplerate,
            blocksize=p.blocksize, dtype='int16')
        self.streamflow.start()

    def run(self):
        '''Callback function'''
        while not self.kill_received:
            _data = self.streamflow.read(p.blocksize)
            self.i_que.put(_data)
        self.i_que.task_done()


class Params:
    def __init__(self):
        self.device = None
        self.channels = 1        # 1 for mono
        self.interval = 30       # minimum time between plot updates
        self.samplerate = 21739  # sampling frequency
        self.winsize = 0.025     # window size including samples to process in sec
        self.winstep = 0.01      # step size (=overlap size) in sec
        self.ncoef = 12          # number of LP coefficients
        self.nfilt = 40          # number of filter banks
        self.num_freq = 1025
        self.preemp = 0.97
        self.update()

    def update(self):
        self.blocksize = int(self.winstep * self.samplerate)
        # self.winsamp = int(self.winsize * self.samplerate)
        self.winsamp = int(self.num_freq // 2 + 1)
        self.nfft = self.winsamp  # number of FFT points


if __name__ == '__main__':
    # Load parameters
    p = Params()  # hyperparameters
    m_params = get_param('model/test', which_epoch=2000)  # model parameters
    phn2idx = np.load('phn2idx.npy').item()
    idx2phn = np.load('idx2phn.npy').item()
    pca = np.load('fft_linfilt_pca.npy').item()
    X_scaler = np.load('fft_linfilt_pca_reduced_scaler.npy').item()
    phones = [*phn2idx]
    # Initialize Feature extractor
    S = SpeechFeatures(None, p.winsize, p.winstep, p.num_freq,
                       p.nfilt, None, win_fun=np.hamming, pre_emp=p.preemp,
                       srate=p.samplerate)
    # Get bins: shape=(nfilt+2,); eg. (42,)
    bins, hz_points = S._make_bins_hz()
    # Initialize queue & buffer
    q = queue.Queue()  # PCM data
    buf = init_buf()   # FIFO buffer
    # Initialize data & plots
    data = dict(linfilt=None, bars=None)
    obj = dict(linfilt=None, bars=None)
    fig, data, obj, txt = init_plot(data, obj)

    # Start audio stream
    stream = Record(q)
    stream.start()

    # Make animation object
    ani = FuncAnimation(fig, update_plot, interval=p.interval, blit=True)
    # Plot!
    try:
        plt.show()
        if not plt.get_fignums():
            stream.kill_received = True
            print('+++ User stopped +++')
    except Exception as e:
        stream.kill_received = True
        print(e)
        print('+++ User stopped +++')
