'''
Speech features library

2018-07-29 Jaekoo Kang
2018-08-02 fixed test functions (librosa, python_speech_feature)

TODO:
- [] Add self.time_vec
- [] Add LPC
- [] Add LSF

Refs:
- https://github.com/jaekookang/useful_bits/blob/master/Speech/Extract_MFCC/Calculate_MFCC.ipynb
- https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
- https://www.kaggle.com/ybonde/log-spectrogram-and-mfcc-filter-bank-example
'''
import ipdb as pdb
import os
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fftpack import dct
import librosa
from python_speech_features import mfcc as pysf_mfcc


class SpeechFeatures:
    '''
    A library for extracting speech features
    Features:
    - FFT
    - FFT (applying linear filters)
    - Mel filterbanks
    - MFCCs
    - LPC
    - LSF
    '''

    def __init__(self, wav_id, win_size, win_step, nfft, nfilt, ndct,
                 win_fun=np.hamming, pre_emp=None):
        # Get variables
        self.wav_id = wav_id
        self.win_size = win_size
        self.win_step = win_step
        self.nfft = nfft
        self.nfilt = nfilt
        self.ndct = ndct
        self.pre_emp = pre_emp
        # Load wav
        self._load_wav()

    def _load_wav(self):
        '''Load wav file'''
        try:
            self.srate, sig = wavfile.read(self.wav_id)
        except:
            # TIMIT has b'NIST' format which throws error
            # when using scipy.io.wavfile.read
            sig, self.srate = librosa.load(self.wav_id, sr=None)
        # Preemphasis
        if self.pre_emp:
            # Check value range (0 ~ 1)
            assert (self.pre_emp <= 1) & (self.pre_emp > 0)
            self.sig = np.append(sig[0], sig[1:] - self.pre_emp * sig[:-1])
        else:
            self.sig = sig

    def _get_frames_windowed(self, sig=None):
        '''Windowing each frame

        TODO:
        - [] Add self.time_vec output
        '''
        self.frame_len = int(self.win_size * self.srate)  # 400
        self.frame_step = int(self.win_step * self.srate)  # 160
        if sig is None:
            sig = self.sig
        sig_len = len(sig)
        # Get number of frames
        #   eg. 359.13 -> 360
        self.num_frames = int(np.ceil(
            np.abs(sig_len - self.frame_len) / self.frame_step))
        # Pad signal
        pad_sig_len = self.num_frames * self.frame_step + self.frame_len  # 57840
        pad = np.zeros((pad_sig_len - sig_len))
        sig_pad = np.append(sig, pad)
        # Get within-frame sample indices
        idx1 = np.tile(
            np.arange(0, self.frame_len), (self.num_frames, 1))  # 360 x 400
        # Get vectors of frame step increments
        idx2 = np.tile(
            np.arange(0, self.num_frames * self.frame_step, self.frame_step),
            ((self.frame_len, 1))).T
        # Get total indices divided by each frame
        indices = idx1 + idx2
        # Get frames divided by each frame based on indices
        frames = sig_pad[indices.astype(np.int32, copy=False)]
        # Windowing
        frames *= np.hamming(self.frame_len)
        return frames

    def get_fft(self, sig=None, nfft=None, linfilt=False, nfilt=None):
        '''Compute SFFT'''
        if nfft is None:
            nfft = self.nfft
        if sig is None:
            sig = self.sig
        # Windowing (returns self.frames)
        self.frames = self._get_frames_windowed(sig=sig)

        # Compute spectrogram
        mag_frames = np.absolute(
            np.fft.rfft(self.frames, n=nfft))  # frames x (NFFT//2+1)
        pow_frames = ((mag_frames)**2)  # squared
        _pow_frames = np.where(pow_frames == 0,
                               np.finfo(float).eps, pow_frames)
        log_frames = 10 * np.log10(_pow_frames)  # See pow2db() in Matlab

        if linfilt:
            if not nfilt:
                nfilt = self.nfilt
            bins, hz_points = self._make_bins_hz(nfilt)
            fbank = self._make_filter_banks(bins, nfft, nfilt)
            # Convolution
            filter_banks = np.dot(pow_frames, fbank.T)
            # Add eps for numerical stability
            filter_banks = np.where(
                filter_banks == 0, np.finfo(float).eps, filter_banks)
            filter_banks = 20 * np.log10(filter_banks)  # to dB
            return filter_banks, fbank
        else:
            return mag_frames, pow_frames, log_frames

    def convert_hz_to_mel(self, hz):
        '''Convert Hz to Mel'''
        return 2595 * np.log10(1 + hz / 700)

    def apply_dct(self, data, ndct=None):
        '''Apply discrete cosine transform
        data: (N, dim); eg. filter_banks
        '''
        # Keep 2nd to (p.ndct+1)th
        # TODO: Find out why the first column is discarded?
        if not ndct:
            ndct = self.ndct
        return dct(data, type=2, axis=1, norm='ortho')[:, 1:(ndct + 1)]

    def get_mfcc(self, pow_frames=None):
        '''Get MFCCs'''
        # Get power spectrum & filter banks
        if pow_frames is None:
            try:
                pow_frames = self.pow_frames
            except:
                pow_frames = None
        filter_banks, fbank = self.get_melfilt(pow_frames=pow_frames)
        # Apply dct
        mfcc = self.apply_dct(filter_banks)
        # Apply simusoidal liftering to de-emphasize higher MFCCs
        (nframes, ncoeff) = mfcc.shape
        n = np.arange(ncoeff)
        cep_lifter = 22  # taken from https://www.kaggle.com/ybonde
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift
        # Mean normalize
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
        return mfcc

    def get_melfilt(self, nfft=None, nfilt=None, pow_frames=None):
        '''Compute Mel filterbanks'''
        if nfft is None:
            nfft = self.nfft
        if nfilt is None:
            nfilt = self.nfilt
        if pow_frames is None:
            _, pow_frames, _ = self.get_fft()

        bins, hz_points = self._make_bins_hz(nfilt, melscale=True)
        fbank = self._make_filter_banks(bins, nfft, nfilt)
        # Convolution
        filter_banks = np.dot(pow_frames, fbank.T)
        # Add eps for numerical stability
        filter_banks = np.where(
            filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)  # to dB
        return filter_banks, fbank

    def _make_bins_hz(self, nfilt=None, melscale=False):
        '''Make bins and hz_points'''
        if not nfilt:
            nfilt = self.nfilt
        # Get hertz scale bins
        if melscale:
            high_freq = (2595 * np.log10(1 + (self.srate / 2) / 700))
            # Get mel points (equally spaced)
            mel_points = np.linspace(0, high_freq, nfilt + 2)
            # Convert Mel to Hz
            hz_points = (700 * (10**(mel_points / 2595) - 1))
        else:
            high_freq = self.srate // 2 + 1
            hz_points = np.linspace(0, high_freq, nfilt + 2)
        # Get bins
        bins = np.floor((self.nfft + 1) * hz_points / self.srate)
        return bins, hz_points

    def _make_filter_banks(self, bins, nfft, nfilt):
        '''Make filter banks'''
        # Filter banks
        fbank = np.zeros((nfilt, int(np.floor(nfft // 2 + 1))))
        # For each filter bank
        for m in range(1, nfilt + 1):  # 1, 2, ... 40
            f_m_minus = int(bins[m - 1])  # left
            f_m = int(bins[m])         # center
            f_m_plus = int(bins[m + 1])  # right

            # For left bins within filter bank
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bins[m - 1]) / \
                    (bins[m] - bins[m - 1])
            # For right bins within filter bank
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
        return fbank

    ###### Librosa #####
    def librosa_mfcc(self, sig=None, ndct=None, nfft=None, nfilt=None):
        '''Compute MFCCs from librosa'''
        if not sig:
            sig, srate = self.sig, self.srate
        if not ndct:
            ndct = self.ndct
        else:
            ndct = 12
        if not nfft:
            nfft = self.nfft
        if not nfilt:
            nfilt = self.nfilt
        # mfcc
        linear = librosa.stft(y=sig, n_fft=nfft, win_length=int(self.win_size*srate),
                              hop_length=int(self.win_step*srate))
        mag = np.abs(linear)  # get magnitude spectrogram
        mel_basis = librosa.filters.mel(srate, nfft, nfilt)
        mel = np.dot(mel_basis, mag)  # n_mel x time
        mel = librosa.amplitude_to_db(mel)
        mfcc = librosa.feature.mfcc(y=None, sr=srate, S=mel, n_mfcc=ndct).T
        # mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
        return mfcc

    ###### python_speech_features #####
    def python_speech_feature_mfcc(self, sig=None, ndct=None, nfilt=None, nfft=None):
        '''Compute MFCCs from python_speech_features'''
        if not sig:
            sig, srate = self.sig, self.srate
        if not ndct:
            ndct = self.ndct
        else:
            ndct = 12
        if not nfilt:
            nfilt = self.nfilt
        if not nfft:
            nfft = self.nfft
        # mfcc
        mfcc = pysf_mfcc(sig, srate, winlen=self.win_size, winstep=self.win_step,
                         numcep=ndct, nfilt=nfilt, nfft=nfft, appendEnergy=False)
        return mfcc

    ###### Debugging tools ######
    def plot(self, data):
        '''Plot raw signal'''
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, ax = plt.subplots(facecolor='white')
        ax.plot(data)
        plt.show()

    def implot(self, data):
        '''Plot 2d data; shape=(N, nfft//2+1)'''
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, ax = plt.subplots(facecolor='white')
        im = ax.imshow(data.T, aspect='auto', origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        plt.show(block=False)


if __name__ == '__main__':
    ### Test this code ###
    wav_id = 'mngu0_s1_0001.wav'
    # Parameters
    pre_emp = 0.97
    win_size = 0.025
    win_step = 0.01
    nfft = 512
    nfilt = 40
    ndct = 36
    # Initialize
    S = SpeechFeatures(wav_id, win_size, win_step, nfft, nfilt, ndct,
                       win_fun=np.hamming, pre_emp=pre_emp)
    # Get FFT spectrogram
    mag_frames, pow_frames, log_frames = S.get_fft()
    # Get FFT spectrogram (linear-filtered)
    filter_banks, filters = S.get_fft(linfilt=True)
    # # Get Mel spectrogram (Mel filterbanks)
    # mel_filter_banks, mel_filters = S.get_melfilt(pow_frames=pow_frames)
    # Get logFFTdct without filterbanks
    FFTDCT = S.apply_dct(log_frames)
    # Get MFCCs
    mfcc = S.get_mfcc()
    mfcc_l = S.librosa_mfcc()
    mfcc_p = S.python_speech_feature_mfcc()
    # S.get_mfcc()
    pdb.set_trace()
