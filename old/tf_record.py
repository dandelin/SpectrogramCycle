import numpy as numpy
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# spectrograms and labels array as input
def convert_to(spectrograms, name):
    num_examples = spectrograms.shape[0]
    rows = spectrograms.shape[1]
    cols = spectrograms.shape[2]
    depth = spectrograms.shape[3] #channel

    print(num_examples, rows, cols, depth)

    filename = name
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        spectrograms_raw = spectrograms[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'num': _int64_feature(num_examples),
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'spectrograms_raw': _bytes_feature(spectrograms_raw)}))
        writer.write(example.SerializeToString())
    # Remember to generate a file name queue of you 'train.TFRecord' file path


def read_and_decode(filename_queue, include_num=False):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    if include_num==True:
        features = tf.parse_single_example(
            serialized_example, \
            features = {
                'spectrograms_raw': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'num': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64)
                })
        spectrograms = tf.decode_raw(features['spectrograms_raw'], tf.float32)
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        num = tf.cast(features['num'], tf.int32)
        depth = tf.cast(features['depth'], tf.int32)
    else:
        features = tf.parse_single_example(
            serialized_example, \
            features = {
                'spectrograms_raw': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64)
                })
        spectrograms = tf.decode_raw(features['spectrograms_raw'], tf.float32)
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        depth = 1
        num = 400

    return spectrograms, height, width, depth, num