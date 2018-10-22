import tensorflow as tf
import numpy as np


def _index_generator(N, batch_size, shuffle=True, seed=None):
    batch_index = 0
    total_batches_seen = 0

    while 1:
        if seed is not None:
            np.random.seed(seed + total_batches_seen)

        current_index = (batch_index * batch_size) % N
        if current_index == 0:
            index_array = np.arange(N)
            if shuffle:
                index_array = np.random.permutation(N)
        if N >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            # current_batch_size = N - current_index
            current_batch_size = batch_size
            batch_index = 0
            current_index = 0
            if shuffle:
                index_array = np.random.permutation(N)
        total_batches_seen += 1

        yield (index_array[current_index: current_index + current_batch_size],
               current_index, current_batch_size)
def tfrecord_read(config,c_dim):

    Filenames = config.tfrecord_train
    crop_patch = config.image_size
    Num_CHANNELS = c_dim
    batch_size = config.Batch_Size
    # output file name string to a queue
    filename_queue = tf.train.string_input_producer([Filenames], num_epochs=None)
    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'low_CompI': tf.FixedLenFeature([], tf.string),
                                           'CompI': tf.FixedLenFeature([], tf.string)
                                       })
    low = tf.decode_raw(features['low_CompI'], tf.float64)
    low = tf.reshape(low, [crop_patch, crop_patch,Num_CHANNELS])

    high = tf.decode_raw(features['CompI'], tf.float64)
    high = tf.reshape(high, [crop_patch, crop_patch,Num_CHANNELS])
    # ###for three echo
    # low = tf.decode_raw(features['low_CompI'], tf.float64)
    # low = tf.reshape(low, [crop_patch_FE, crop_patch_PE,6])
    # low = low[:,:,3:6]
    #
    # high = tf.decode_raw(features['CompI'], tf.float64)
    # high = tf.reshape(high, [crop_patch_FE, crop_patch_PE,6])
    # high = high[:,:,3:6]
    ####
    low_batch, high_batch = tf.train.shuffle_batch([low, high], batch_size, capacity=20000, min_after_dequeue=5000)
    low_image = tf.reshape(low_batch,[batch_size,crop_patch, crop_patch,Num_CHANNELS])
    high_image = tf.reshape(high_batch,[batch_size,crop_patch, crop_patch,Num_CHANNELS])


    return low_image,high_image