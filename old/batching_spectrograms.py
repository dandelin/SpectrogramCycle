import tensorflow as tf
import numpy as np
import librosa
from glob import glob
from tf_record import read_and_decode

def wh_from_sample(dirname, sess=None):
    specs = glob('{}/*.spec'.format(dirname))
    filename_queue = tf.train.string_input_producer(list(specs), shuffle=True)
    spectrogram, height, width, depth, num = read_and_decode(filename_queue)
    if sess==None:
        sess = tf.InteractiveSession()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return sess.run(height), sess.run(width)

def spec_batch(dirname, sess=None, batch_size=1, num_threads=4, min_after_dequeue=100, shuffle=True):
    specs = glob('{}/*.spec'.format(dirname))

    filename_queue = tf.train.string_input_producer(list(specs), shuffle=True)
    spectrogram, height, width, depth, num = read_and_decode(filename_queue)

    height, width = wh_from_sample(dirname)
    print(height, width)
    spectrogram = tf.reshape(spectrogram, [height, width, 1])

    print(spectrogram)

    # spectrogram /= tf.reduce_max(spectrogram)
    # spectrogram *= 255
    # spectrogram = spectrogram / 127.5 - 1

    if shuffle==True:
        capacity = min_after_dequeue + (num_threads + 1) * batch_size
        spec_batch = tf.train.shuffle_batch([spectrogram], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue, num_threads=num_threads)
    else:
        spec_batch = tf.train.batch([spectrogram])

    if sess==None:
        sess = tf.InteractiveSession()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #print(sess.run(spec_batch))
    #save_reconstructed_audio(sess.run(spec_batch), 're.wav')

    return spec_batch, num

class SpecData:

    def __init__(self, session, dirname, batch_size, retain_all=False):
        self.sess = session
        if retain_all==False:
            self.spec_batch, self.num = spec_batch(dirname, sess=session, batch_size=batch_size, shuffle=True)
        else:
            self.spec_batch, self.num = spec_batch(dirname, sess=session, batch_size=batch_size, shuffle=False)

    def __len__(self):
        return self.num

    def batch_ops(self):
        return self.spec_batch

    def batch(self):
        return self.sess.run(self.spec_batch)


# Griffin lim algorithm (REAL --> COMPLEX)
# save_reconstructed_audio(s[:,:1000], 're.wav')
# S[spectrum range, 100 per sec]
def save_reconstructed_audio(spectrogram, filename, iter=100):
    spectrogram = np.reshape(spectrogram, [spectrogram.shape[1], spectrogram.shape[2]])
    SAMPLING_RATE = 16000
    FFT_SIZE = 1022 #Frequency resolution
    HOP_LENGTH = None
    # spectrogram = np.power(10, spectrogram) # If the input specrogram is scaled with logarithm, use this line.
    p = 2 * np.pi * np.random.random_sample(spectrogram.shape) - np.pi
    for i in range(iter):
        # S = ((spectrogram + 1) * 127.5) * np.exp(1j*p)
        S = spectrogram * np.exp(1j*p)
        x = librosa.istft(S, hop_length = HOP_LENGTH, win_length = FFT_SIZE)
        p = np.angle(librosa.stft(x, n_fft = FFT_SIZE, hop_length = HOP_LENGTH))
    librosa.output.write_wav(filename, x, SAMPLING_RATE)

if __name__=='__main__':
    spec_batch('datasets/bush', None)