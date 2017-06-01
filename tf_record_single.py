import numpy as numpy
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# spectrograms and labels array as input
def convert_to(spectrogram, name):
    rows = spectrogram.shape[0]
    cols = spectrogram.shape[1]
    depth = spectrogram.shape[2] #channel

    print(rows, cols, depth)

    filename = name
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    spectrogram_raw = spectrogram.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'spectrogram_raw': _bytes_feature(spectrogram_raw)}))
    writer.write(example.SerializeToString())

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        'spectrogram_raw': tf.FixedLenFeature([], tf.string),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64)
        })
    spectrograms = tf.decode_raw(features['spectrogram_raw'], tf.float32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)

    return spectrograms, height, width, depth