from keras import backend as K
import tensorflow as tf
from keras.layers import LSTM, Merge, Dropout, Input, Dense, Bidirectional
from keras import metrics as m
from keras import models
from keras import optimizers
import librosa
from glob import glob
from . import features
import numpy as np
import h5py as h5py

class Model(object):
    def __init__ (self, hdf5=None, audio=None):
        self.hdf5 = hdf5
        self.audio = audio
        self.sampling_rate = 20480
        self.sample_size_sec = 5
        self.window_size = int(self.sampling_rate * 0.2)
        self.window_step = int(self.sampling_rate * 0.1)
        self._model = models.load_model(hdf5) if hdf5 else None
        self.sound_clip, _ = librosa.load(self.audio, sr = self.sampling_rate) if self.audio else (None, None)

    def setModel (self, hdf5):
        self.hdf5 = hdf5
        self._model = models.load_model(hdf5)

    def setAudio (self, audio):
        self.audio = audio
        self.sound_clip, _ = librosa.load(audio, sr = self.sampling_rate)

    def getLevel (self, audio=None):
        window = self.sampling_rate * self.sample_size_sec
        l = len(self.sound_clip)//window
        start = 0
        end = l*window
        res = []
        for i in range(start, end, window):
            res.append(features.extract_feature(self.sound_clip[i:i+window], self.sampling_rate, 12, self.window_size, self.window_step))
        return np.round(self._model.predict(np.array(res)), 1)