import tensorflow as tf
import numpy as np
import h5py
from scipy.misc import imsave
import tensorlayer as tl
from SRCNN_configs import config

# ###data for one_channel
# crop_number = 50
# crop_patch_size = 80
# NUM_CHANNELS = config.NUM_CHANNELS
# crop_size_PE = config.crop_size_PE
# crop_size_FE = config.crop_size_FE
# tfrecord_filename = config.tfrecord_filename
# PE_size_ori =config.PE_size_ori
# FE_size_ori =config.FE_size_ori
# batch_x= np.zeros((1,FE_size_ori,PE_size_ori,NUM_CHANNELS))
# batch_y = np.zeros((1,FE_size_ori,PE_size_ori,NUM_CHANNELS))
#
# writer = tf.python_io.TFRecordWriter(tfrecord_filename)
# s1 = h5py.File("train"  + '/low_CompI_final.mat')
# X_data1 = s1['low_CompI_final'].value
# X_data1 = np.transpose(X_data1, [2, 1, 0])###nFE,nPE,nSL
# X_data = X_data1[:,:,:]
#
# s1 = h5py.File("train"  + '/CompI_final.mat')
# Y_data1 = s1['CompI_final'].value
# Y_data1 = np.transpose(Y_data1, [ 2, 1, 0])
# Y_data = Y_data1[:,:,:]
# nb_images = Y_data.shape[2]
#
# k=0
# for j in range(nb_images):
#     for i in range(crop_number):
#         X_input = X_data[:,:,j].reshape(1,crop_size_FE,crop_size_PE,1)
#         Y_input = Y_data[:,:,j].reshape(1,crop_size_FE,crop_size_PE,1)
#         Input = np.concatenate((X_input, Y_input))
#         Output = tl.prepro.crop_multi(Input,crop_patch_size,crop_patch_size,True)
#         X_small_patch = Output[0,:,:,0].tostring()
#         Y_small_patch = Output[1,:,:,0].tostring()
#         example = tf.train.Example(
#             features=tf.train.Features(
#                 feature={'low_CompI': tf.train.Feature(bytes_list=tf.train.BytesList(value=[X_small_patch])),
#                          'CompI': tf.train.Feature(bytes_list=tf.train.BytesList(value=[Y_small_patch]))
#                          }
#             )
#         )
#         # X_small_patch[k:,:,0],Y_small_patch[k,:,:,0]  = tl.prepro.crop_multi([X_input, Y_input],16,16,True)
#         serialized = example.SerializeToString()
#         writer.write(serialized)
# writer.close()


# ###data for six_channel
# crop_number = 50
# crop_patch_size = 80
# patch_size_PE = 288
# patch_size_FE = 384
# batch_x= np.zeros((1,patch_size_PE,patch_size_FE,6))
# batch_y = np.zeros((1,patch_size_PE,patch_size_FE,6))
# #
# writer = tf.python_io.TFRecordWriter('Amp_6channel.tfrecord')
# #
# # all echo in one mat
# s1 = h5py.File("train//Amp_6echo"  + '/low_CompI_final.mat')
# X_data1 = s1['low_CompI_final'].value
# X_data = X_data1[:,:,:,:]
#
# s1 = h5py.File("train//Amp_6echo"  + '/CompI_final.mat')
# Y_data1 = s1['CompI_final'].value
# Y_data = Y_data1[:,:,:,:]
# nb_images = Y_data.shape[1]
# # X_small_patch = np.zeros((nb_images*crop_number,crop_patch_size,crop_patch_size,1))
# # Y_small_patch = np.zeros((nb_images*crop_number,crop_patch_size,crop_patch_size,1))
# image_shape = (crop_patch_size, crop_patch_size,1)
# y_image_shape = (crop_patch_size, crop_patch_size,1)
# k=0
# for j in range(nb_images):
#     for i in range(crop_number):
#         batch_x[0, :, :, 0] = X_data[0,j, :, :].astype('float32')
#         batch_x[0, :, :, 1] = X_data[1,j, :, :].astype('float32')
#         batch_x[0, :, :, 2] = X_data[2,j, :, :].astype('float32')
#         batch_x[0, :, :, 3] = X_data[3,j, :, :].astype('float32')
#         batch_x[0, :, :, 4] = X_data[4,j, :, :].astype('float32')
#         batch_x[0, :, :, 5] = X_data[5,j, :, :].astype('float32')
#
#         batch_y[0, :, :, 0] = Y_data[0,j, :, :].astype('float32')
#         batch_y[0, :, :, 1] = Y_data[1,j, :, :].astype('float32')
#         batch_y[0, :, :, 2] = Y_data[2,j, :, :].astype('float32')
#         batch_y[0, :, :, 3] = Y_data[3,j, :, :].astype('float32')
#         batch_y[0, :, :, 4] = Y_data[4,j, :, :].astype('float32')
#         batch_y[0, :, :, 5] = Y_data[5,j, :, :].astype('float32')
#         Input = np.concatenate((batch_x, batch_y))
#         # ##### crop
#         Output = tl.prepro.crop_multi(Input,crop_patch_size,crop_patch_size,True)
#         X_small_patch = Output[0,:,:,0:6].tostring()
#         Y_small_patch = Output[1,:,:,0:6].tostring()
#         example = tf.train.Example(
#             features=tf.train.Features(
#                 feature={'low_CompI': tf.train.Feature(bytes_list=tf.train.BytesList(value=[X_small_patch])),
#                          'CompI': tf.train.Feature(bytes_list=tf.train.BytesList(value=[Y_small_patch]))
#                          }
#             )
#         )
#         # X_small_patch[k:,:,0],Y_small_patch[k,:,:,0]  = tl.prepro.crop_multi([X_input, Y_input],16,16,True)
#         serialized = example.SerializeToString()
#         writer.write(serialized)
# writer.close()
######### Address for 12channel data(complex)
crop_number = 50
crop_patch_size = 80
NUM_CHANNELS = config.NUM_CHANNELS
crop_size_PE = config.crop_size_PE
crop_size_FE = config.crop_size_FE
tfrecord_filename = config.tfrecord_filename
PE_size_ori =config.PE_size_ori
FE_size_ori =config.FE_size_ori
batch_x= np.zeros((1,FE_size_ori,PE_size_ori,NUM_CHANNELS))
batch_y = np.zeros((1,FE_size_ori,PE_size_ori,NUM_CHANNELS))
#
writer = tf.python_io.TFRecordWriter(tfrecord_filename)
s1 = h5py.File("train"  + '/low_CompI_final.mat')
X_data1 = s1['low_CompI_final'].value
X_data1 = np.transpose(X_data1, [3, 2, 1, 0])###nFE,nPE,nSL,nCH
X_data = X_data1[:,:,:,:]

s1 = h5py.File("train"  + '/CompI_final.mat')
Y_data1 = s1['CompI_final'].value
Y_data1 = np.transpose(Y_data1, [3, 2, 1, 0])
Y_data = Y_data1[:,:,:,:]
nb_images = Y_data.shape[2]
# X_small_patch = np.zeros((nb_images*crop_number,crop_patch_size,crop_patch_size,1))
# Y_small_patch = np.zeros((nb_images*crop_number,crop_patch_size,crop_patch_size,1))

k=0
for j in range(nb_images):
    for i in range(crop_number):
        #
        batch_y[0, :, :, 0] = Y_data[:, :, j, 0].astype('float32')
        batch_y[0, :, :, 1] = Y_data[:, :, j, 1].astype('float32')
        batch_y[0, :, :, 2] = Y_data[:, :, j, 2].astype('float32')
        batch_y[0, :, :, 3] = Y_data[:, :, j, 3].astype('float32')
        batch_y[0, :, :, 4] = Y_data[:, :, j, 4].astype('float32')
        batch_y[0, :, :, 5] = Y_data[:, :, j, 5].astype('float32')
        # batch_y[0, :, :, 6] = Y_data[:, :, j, 6].astype('float32')
        # batch_y[0, :, :, 7] = Y_data[:, :, j, 7].astype('float32')
        # batch_y[0, :, :, 8] = Y_data[:, :, j, 8].astype('float32')
        # batch_y[0, :, :, 9] = Y_data[:, :, j, 9].astype('float32')
        # batch_y[0, :, :, 10] = Y_data[:, :, j, 10].astype('float32')
        # batch_y[0, :, :, 11] = Y_data[:, :, j, 11].astype('float32')

        batch_x[0, :, :, 0] = X_data[:, :, j, 0].astype('float32')
        batch_x[0, :, :, 1] = X_data[:, :, j, 1].astype('float32')
        batch_x[0, :, :, 2] = X_data[:, :, j, 2].astype('float32')
        batch_x[0, :, :, 3] = X_data[:, :, j, 3].astype('float32')
        batch_x[0, :, :, 4] = X_data[:, :, j, 4].astype('float32')
        batch_x[0, :, :, 5] = X_data[:, :, j, 5].astype('float32')
        # batch_x[0, :, :, 6] = X_data[:, :, j, 6].astype('float32')
        # batch_x[0, :, :, 7] = X_data[:, :, j, 7].astype('float32')
        # batch_x[0, :, :, 8] = X_data[:, :, j, 8].astype('float32')
        # batch_x[0, :, :, 9] = X_data[:, :, j, 9].astype('float32')
        # batch_x[0, :, :, 10] = X_data[:, :, j, 10].astype('float32')
        # batch_x[0, :, :, 11] = X_data[:, :, j, 11].astype('float32')
        Input = np.concatenate((batch_x, batch_y))
        # ##### crop
        Output = tl.prepro.crop_multi(Input,crop_size_PE,crop_size_FE,True)
        # Output = tl.prepro.elastic_transform_multi(Output, alpha=255*3, sigma=255*0.10, is_random=True)
        # Output = tl.prepro.flip_axis_multi(Output, axis=1, is_random=True)
        # Output = tl.prepro.rotation_multi(Output,rg=10, is_random=True, fill_mode='constant')
        # Output = tl.prepro.shift_multi(Output, wrg=0.10, hrg=0.10, is_random=True,fill_mode='constant')
        # Output = tl.prepro.zoom_multi(Output,zoom_range=[0.90,1.10],is_random=True, fill_mode='constant')
        # Output = tl.prepro.brightness_multi(Output,gamma=0.05, is_random=True)
        X_small_patch = Output[0,:,:,0:NUM_CHANNELS].tostring()
        Y_small_patch = Output[1,:,:,0:NUM_CHANNELS].tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={'low_CompI': tf.train.Feature(bytes_list=tf.train.BytesList(value=[X_small_patch])),
                         'CompI': tf.train.Feature(bytes_list=tf.train.BytesList(value=[Y_small_patch]))
                         }
            )
        )
        # X_small_patch[k:,:,0],Y_small_patch[k,:,:,0]  = tl.prepro.crop_multi([X_input, Y_input],16,16,True)
        serialized = example.SerializeToString()
        writer.write(serialized)
writer.close()


# output file name string to a queue
# ## 测试单通道的tfrecord数据
# filename_queue = tf.train.string_input_producer(['Amp_all_echo_one_channel.tfrecord'], num_epochs=None)
# # create a reader from file queue
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(filename_queue)
# # get feature from serialized example
# features = tf.parse_single_example(serialized_example,
#                                    features={
#                                        'low_CompI': tf.FixedLenFeature([],tf.string),
#                                         'CompI': tf.FixedLenFeature([],tf.string)
#                                    })
# low = tf.decode_raw(features['low_CompI'],tf.float64)
# low = tf.reshape(low,[crop_patch_size,crop_patch_size])
#
# high = tf.decode_raw(features['CompI'], tf.float64)
# high = tf.reshape(high, [crop_patch_size, crop_patch_size])
# low_batch, high_batch = tf.train.shuffle_batch([low, high],batch_size =16, capacity= 50000, min_after_dequeue=20000, num_threads=1)
# high_batch = tf.reshape(high_batch, [16,crop_patch_size,crop_patch_size,1])
#
# sess = tf.Session()
# init = tf.initialize_all_variables()
# sess.run(init)
#
# tf.train.start_queue_runners(sess=sess)
# for i in range(100):
#     low_images, high_images = sess.run([low_batch, high_batch])
#     d = high_images[0,:,:,0]
#     imsave('high.bmp',d*255)


# ###测试六通道的tfrecord数据
filename_queue = tf.train.string_input_producer(['Amp_12channel.tfrecord'], num_epochs=None)
# create a reader from file queue
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# get feature from serialized example
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'low_CompI': tf.FixedLenFeature([],tf.string),
                                        'CompI': tf.FixedLenFeature([],tf.string)
                                   })
low = tf.decode_raw(features['low_CompI'],tf.float64)
low = tf.reshape(low,[crop_patch_size,crop_patch_size,12])

high = tf.decode_raw(features['CompI'], tf.float64)
high = tf.reshape(high, [crop_patch_size, crop_patch_size,12])
low_batch, high_batch = tf.train.shuffle_batch([low, high],batch_size =16, capacity= 50000, min_after_dequeue=20000, num_threads=1)
high_batch = tf.reshape(high_batch, [16,crop_patch_size,crop_patch_size,12])

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

tf.train.start_queue_runners(sess=sess)
for i in range(100):
    low_images, high_images = sess.run([low_batch, high_batch])
    d = high_images[0,:,:,0]
    imsave('high.bmp',d*255)