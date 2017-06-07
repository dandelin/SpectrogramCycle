import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
from os import walk, mkdir
from glob import glob
from tf_record_single import convert_to

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_spectrograms_from_mp3s(dirpath):
    SAMPLING_RATE = 16000
    FFT_SIZE = 1022 #Frequency resolution
    HOP_LENGTH = None # int(FFT_SIZE/6.5)
    DURATION = 8.15

    audios = glob(dirpath + '/*.mp3')

    for i, audio in enumerate(audios):
        print("{}th audio : ".format(i), audio)
        y, sr = librosa.core.load(audio, sr=SAMPLING_RATE, mono=True)
        length = int(librosa.get_duration(y) / DURATION)
        for j, cutpoint in enumerate(range(length)):
            print("{}/{}".format(j, length))
            block = y[cutpoint * SAMPLING_RATE:int((cutpoint + DURATION) * SAMPLING_RATE)]
            D = librosa.stft(block, n_fft=FFT_SIZE)
            D = np.abs(D)   # since D is a complex number
            D = np.expand_dims(D, axis=2)
            name = audio[:-4] + '{}.spec'.format(j)
            convert_to(D, name)

if __name__=='__main__':
    make_spectrograms_from_mp3s('datasets/park')