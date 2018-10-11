from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.misc import imsave, imread, imresize
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
from PIL import Image
import csv
import time
import tensorlayer as tl
from SRCNN_configs import *
import scipy.io as sio
import h5py


patch_size =128

'将图像分割成若干小块'
def transform_images(directory_low,directory_high, output_directory_low, output_directory_high, img_size, patch_size, stride, csv_datatype = 1, max_nb_images=-1):
    index = 1
    index_file = 1
    imgarray_x = []
    imgarray_y = []
    # For each image in input_images directory
    orginal_num_images_x = len([name for name in os.listdir(directory_low)])

    if max_nb_images != -1:
        print("Transforming %d images." % max_nb_images)
    else:
        assert max_nb_images <= orginal_num_images_x, "Max number of images must be less than number of images in path"
        print("Transforming %d images." % (orginal_num_images_x))



    for file in os.listdir(directory_low):
        #img = imread(directory + file)


        img_low = Image.open(directory_low + file)
        img_low = np.asarray(img_low)
        img_high = Image.open(directory_high+file)
        img_high = np.asarray(img_high)
        for dis in range(10):
            img_distor_x, img_distor_y = distort_imgs([img_low, img_high],img_size)
            # Create patches
            nb_images = (((img_size-patch_size) // stride + 1) ** 2)

            hr_samples_low = np.empty((nb_images, patch_size, patch_size,1))
            hr_samples_high = np.empty((nb_images, patch_size, patch_size,1))


            image_subsample_iterator_low = subimage_generator(img_distor_x, img_size, stride, patch_size, nb_images)
            image_subsample_iterator_high = subimage_generator(img_distor_y, img_size, stride, patch_size, nb_images)

            stride_range = np.sqrt(nb_images).astype(int)

            i = 0
            for j in range(stride_range):
                for k in range(stride_range):
                    hr_samples_low[i] = next(image_subsample_iterator_low)
                    hr_samples_high[i] = next(image_subsample_iterator_high)
                    i += 1


            t1 = time.time()
            # Create nb_hr_images 'X' and 'Y' sub-images of size hr_patch_size for each patch
            for i in range(nb_images):
                if(csv_datatype == 0):
                #保存为图像
                    ip_low = hr_samples_low[i].reshape((patch_size,patch_size))
                    imsave(output_directory_high + "%d_%d_%d.bmp" % (index_file, i, dis + 1), ip_low)
                    ip_high = hr_samples_high[i].reshape((patch_size,patch_size))
                    imsave(output_directory_low + "%d_%d_%d.bmp" % (index_file, i, dis + 1),ip_high)
                    index += 1
                else:
                    #保存为csv
                    ip_low = hr_samples_low[i]
                    ip_high = hr_samples_high[i]

                    '''将数据归一化为0到1范围 '''
                    # scaler = MinMaxScaler(feature_range=(0,1))
                    # ip_rescale = scaler.fit_transform(ip)

                    '''将数据normalize化 '''
                    # scaler = Normalizer().fit(ip)
                    # ip_Norm= scaler.transform(ip)
                    ip_ndarray_low = np.ndarray.flatten(ip_low)#压平为一维向量
                    ip_ndarray1_low = ip_ndarray_low.tolist()
                    imgarray_x.append(ip_ndarray1_low)

                    ip_ndarray_high = np.ndarray.flatten(ip_high)#压平为一维向量
                    ip_ndarray1_high = ip_ndarray_high.tolist()
                    imgarray_y.append(ip_ndarray1_high)
                    if (index % 100 == 0) or (index == orginal_num_images_x):
                        with open(output_directory_high + 'train_Y.csv', 'a', newline='') as myfile:
                            mywriter = csv.writer(myfile)
                            mywriter.writerows(imgarray_y)
                            myfile.close()
                        with open(output_directory_low + 'train_X.csv', 'a', newline='') as myfile:
                            mywriter = csv.writer(myfile)
                            mywriter.writerows(imgarray_x)
                            myfile.close()
                        imgarray_x = []
                        imgarray_y = []
                    index += 1



        print("Finished distortion image %d of image %d in time %0.2f seconds and index %d. (%s)" % (dis, index_file, time.time() - t1, index, file))
        index_file += 1

    print("Images transformed. Saved at directory : %s, %s" % (output_directory_low, output_directory_high))

def subimage_generator(img, img_size, stride, patch_size, nb_hr_images):
    for _ in range(nb_hr_images):
        for x in range(0, img_size - patch_size+1, stride):
            for y in range(0, img_size - patch_size+1, stride):
                subimage = img[x: x + patch_size, y: y + patch_size]

                yield subimage

def image_generator(directory, patch_size, batch_size, shuffle=True, seed=None):

    image_shape = (patch_size, patch_size,1)
    y_image_shape = (patch_size, patch_size,1)

    # file_names = [f for f in sorted(os.listdir(directory + "high/"))]
    # X_filenames = [os.path.join(directory, "low", f) for f in file_names]
    # y_filenames = [os.path.join(directory, "high", f) for f in file_names]
    file_names = [f for f in sorted(os.listdir(directory + "CompI_echo1_FA1/"))]
    X_filenames = [os.path.join(directory, "low_CompI_echo1_FA1", f) for f in file_names]
    y_filenames = [os.path.join(directory, "CompI_echo1_FA1", f) for f in file_names]
    nb_images = len(file_names)
    #print("Found %d images." % nb_images)

    index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + image_shape)
        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):
            x_fn = X_filenames[j]
            img = imread(x_fn)
            img = img.astype('float32') / 255.
            batch_x[i,:,:,0] = img
            y_fn = y_filenames[j]
            img = imread(y_fn)
            img = img.astype('float32') / 255.
            batch_y[i,:,:,0] = img

        yield (batch_x, batch_y)

def image_generator_complex(directory, patch_size, batch_size, shuffle=True, seed=None):

    image_shape = (patch_size, patch_size,2)
    y_image_shape = (patch_size, patch_size,2)

    file_names = [f for f in sorted(os.listdir(directory + "X_real/"))]
    X_filenames_real = [os.path.join(directory, "X_real", f) for f in file_names]
    X_filenames_imag = [os.path.join(directory, "X_imag", f) for f in file_names]

    y_filenames_real = [os.path.join(directory, "Y_real", f) for f in file_names]
    y_filenames_imag = [os.path.join(directory, "Y_imag", f) for f in file_names]

    nb_images = len(file_names)
    #print("Found %d images." % nb_images)

    index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + image_shape)

        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):
            x_fn1 = X_filenames_real[j]
            batch_x[i, :, :, 0] = imread(x_fn1)
            x_fn2 = X_filenames_imag[j]
            batch_x[i,:,:,1] = imread(x_fn2)
            batch_x[i,:,:,:] = batch_x[i,:,:,:].astype('float32') / 255


            y_fn1 = y_filenames_real[j]
            batch_y[i, :, :, 0] = imread(y_fn1)
            y_fn2 = y_filenames_imag[j]
            batch_y[i, :, :, 1] = imread(y_fn2)
            batch_y[i, :, :, :] = batch_y[i,:,:,:].astype('float32') / 255

        yield (batch_x, batch_y)

# ### mat read
def transform_images_mat(directory, patch_size, scale, batch_size, shuffle=True, seed=None, crop_number=50,crop_patch_size= 64):
    s1 = h5py.File(directory+'/low_CompI.mat')
    X_dataa = s1['low_CompI'].value
    X_data = X_dataa[:,:,:]
    s2 = h5py.File(directory+'/CompI.mat')
    Y_dataa = s2['CompI'].value
    Y_data = Y_dataa[:, :, :]
    nb_images = Y_data.shape[0]
    X_small_patch = np.zeros((nb_images*crop_number,crop_patch_size,crop_patch_size,1))
    Y_small_patch = np.zeros((nb_images*crop_number,crop_patch_size,crop_patch_size,1))
    image_shape = (crop_patch_size, crop_patch_size,1)
    y_image_shape = (crop_patch_size, crop_patch_size,1)
    k=0
    for j in range(nb_images):
        for i in range(crop_number):
            X_input = X_data[j,:,:].reshape(1,patch_size,patch_size,1)
            Y_input = Y_data[j,:,:].reshape(1,patch_size,patch_size,1)
            Input = np.concatenate((X_input, Y_input))
            Output = tl.prepro.crop_multi(Input,crop_patch_size,crop_patch_size,True)
            X_small_patch[k, :, :, 0] = Output[0,:,:,0]
            Y_small_patch[k, :, :, 0] = Output[1,:,:,0]
            # X_small_patch[k:,:,0],Y_small_patch[k,:,:,0]  = tl.prepro.crop_multi([X_input, Y_input],16,16,True)

            k +=1
    index_generator = _index_generator(nb_images*crop_number, batch_size, shuffle, seed)
    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + image_shape)

        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):
            # X_data = scipy.misc.imresize(Y_data[j,:,:],(patch_size,patch_size))
            # batch_x[i, :, :, 0] = X_data

            # batch_x[i, :, :, 0] = X_data[j,:,:].astype('float32')
            # batch_y[i, :, :, 0] = Y_data[j,:,:].astype('float32')
            #
            #
            # batch_x[i, :, :, 0] = X_data
            # batch_y[i, :, :, 0] = Y_data
            batch_x[i, :, :, 0] = X_small_patch[j,:,:,0].astype('float32')
            batch_y[i, :, :, 0] = Y_small_patch[j,:,:,0].astype('float32')
        yield (batch_x, batch_y)
def image_generator_mat(directory, patch_size_PE, patch_size_FE, scale, batch_size, shuffle=True, seed=None):

    image_shape = (patch_size_PE, patch_size_FE,1)
    y_image_shape = (patch_size_PE*scale, patch_size_FE*scale,1)


    #
    # s1 = h5py.File(directory+'/low_CompI_echo1_FA1_for_multi_channel.mat')
    # X_dataa = s1['low_CompI_echo1_FA1_for_multi_channel'].value.astype(np.uint8)
    # X_data = X_dataa[0:48,:,:]
    #
    #
    #
    # s2 = h5py.File(directory+'/CompI_echo1_FA1_for_multi_channel.mat')
    # Y_dataa = s2['CompI_echo1_FA1_for_multi_channel'].value.astype(np.uint8)
    # # Y_dataa = s2['CompI'].value
    # Y_data = Y_dataa[0:48,:,:]


    # s1 = h5py.File(directory+'/low_CompI_echo1_FA1_for_multi_channel_zero_padding.mat')
    # X_dataa = s1['low_CompI_echo1_FA1_for_multi_channel_zero_padding'].value
    # X_data = X_dataa[:,:,:]
    #
    #
    #
    # s2 = h5py.File(directory+'/CompI_echo1_FA1_for_multi_channel.mat')
    # Y_dataa = s2['CompI_echo1_FA1_for_multi_channel'].value
    # Y_data = Y_dataa[:, :, :]

    s1 = h5py.File(directory+'/low_CompI_final.mat')
    X_dataa = s1['low_CompI_final'].value
    X_dataa = np.transpose(X_dataa,[2,1,0])
    X_data = np.abs(X_dataa)



    s2 = h5py.File(directory+'/CompI_final.mat')
    Y_dataa = s2['CompI_final'].value
    Y_dataa = np.transpose(Y_dataa,[2,1,0])
    Y_data = Y_dataa[:, :, :]
    # for s in range(10):

    nb_images = Y_data.shape[2]
    # nb_images =10

    index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + image_shape)

        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):
            # X_data = scipy.misc.imresize(Y_data[j,:,:],(patch_size,patch_size))
            # batch_x[i, :, :, 0] = X_data

            batch_x[i, :, :, 0] = X_data[:,:,j]
            batch_y[i, :, :, 0] = Y_data[:,:,j]

            #
            # batch_x[i, :, :, 0] = X_data
            # batch_y[i, :, :, 0] = Y_data

        yield (batch_x, batch_y)


def three_channel_image_generator_mat(directory, patch_size_PE,patch_size_FE, scale, batch_size, shuffle=True,
                                          seed=None):

    image_shape = (patch_size_PE, patch_size_FE, 3)
    y_image_shape = (patch_size_PE * scale, patch_size_FE * scale, 3)

    # s1 = sio.loadmat(directory+'/low_CompI_echo1.mat')
    # s1_list = list(s1.keys())
    # X_data = s1[s1_list[-1]]
    # s2 = sio.loadmat(directory+'/CompI_echo1.mat')
    # s2_list = list(s2.keys())
    # Y_data = s2[s2_list[-1]]


    #     nb_images = X_data_echo1_FA1.shape[0]
    # all echo in one mat
    # s1 = h5py.File(directory + '/low_CompI_final.mat')
    # X_data1 = s1['low_CompI_final'].value
    # X_data = X_data1[:,:,:,:]
    #
    # s1 = h5py.File(directory + '/CompI_final.mat')
    # Y_data1 = s1['CompI_final'].value
    # Y_data = Y_data1[:,:,:,:]
    # nb_images = Y_data.shape[1]
    # s1 = h5py.File(directory + '/low_CompI_final.mat')
    # X_data1 = s1['low_CompI_final'].value
    # X_data1 = np.transpose(X_data1, [3, 2, 1, 0])###nFE,nPE,nSL,nCH
    # X_data = X_data1[:, :, :, :]
    #
    # s1 = h5py.File(directory + '/CompI_final.mat')
    # Y_data1 = s1['CompI_final'].value
    # Y_data1 = np.transpose(Y_data1, [3, 2, 1, 0])
    # Y_data = Y_data1[:, :, :, :]
    # nb_images = Y_data.shape[2]
    # nb_images = 1
### for nPE*nSL, nCh, nFE*n_Volunteer
    s1 = h5py.File(directory + '/low_CompI_final.mat')
    X_data1 = s1['low_CompI_final'].value
    X_data1 = np.transpose(X_data1, [3, 2, 1, 0])###nPE,nSL,nFE*9,nCH
    X_data = X_data1[:, :, :, 3:6]

    s1 = h5py.File(directory + '/CompI_final.mat')
    Y_data1 = s1['CompI_final'].value
    Y_data1 = np.transpose(Y_data1, [3, 2, 1, 0])
    Y_data = Y_data1[:, :, :, 3:6]
    nb_images = Y_data.shape[2]



    # print("Found %d images." % nb_images)

    index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + image_shape)

        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):

            # batch_x[i, :, :, 0] = X_data_echo1_FA1[j, :, :].astype('float32')
            # batch_x[i, :, :, 1] = X_data_echo2_FA1[j, :, :].astype('float32')
            # batch_x[i, :, :, 2] = X_data_echo3_FA1[j, :, :].astype('float32')
            # batch_x[i, :, :, 3] = X_data_echo1_FA2[j, :, :].astype('float32')
            # batch_x[i, :, :, 4] = X_data_echo2_FA2[j, :, :].astype('float32')
            # batch_x[i, :, :, 5] = X_data_echo3_FA2[j, :, :].astype('float32')
            # batch_x[i, :, :, 0] = X_data[0,j, :, :].astype('float32')
            # batch_x[i, :, :, 1] = X_data[1,j, :, :].astype('float32')
            # batch_x[i, :, :, 2] = X_data[2,j, :, :].astype('float32')
            # batch_x[i, :, :, 3] = X_data[3,j, :, :].astype('float32')
            # batch_x[i, :, :, 4] = X_data[4,j, :, :].astype('float32')
            # batch_x[i, :, :, 5] = X_data[5,j, :, :].astype('float32')
            batch_y[i, :, :, 0] = Y_data[:,:,j,0].astype('float32')
            batch_y[i, :, :, 1] = Y_data[:,:,j,1].astype('float32')
            batch_y[i, :, :, 2] = Y_data[:,:,j,2].astype('float32')


            batch_x[i, :, :, 0] = X_data[:,:,j,0].astype('float32')
            batch_x[i, :, :, 1] = X_data[:,:,j,1].astype('float32')
            batch_x[i, :, :, 2] = X_data[:,:,j,2].astype('float32')

            # batch_y[i, :, :, 0] = Y_data[0,j, :, :].astype('float32')
            # batch_y[i, :, :, 1] = Y_data[1,j, :, :].astype('float32')
            # batch_y[i, :, :, 2] = Y_data[2,j, :, :].astype('float32')
            # batch_y[i, :, :, 3] = Y_data[3,j, :, :].astype('float32')
            # batch_y[i, :, :, 4] = Y_data[4,j, :, :].astype('float32')
            # batch_y[i, :, :, 5] = Y_data[5,j, :, :].astype('float32')
            # batch_x[i,:,:,0] = X_data_echo1_FA1.astype('float32')
            # batch_x[i,:,:,1] = X_data_echo2_FA1.astype('float32')
            # batch_x[i,:,:,2] = X_data_echo3_FA1.astype('float32')
            # batch_y[i,:,:,0] = Y_data_echo1_FA1.astype('float32')
            # batch_y[i,:,:,1] = Y_data_echo2_FA1.astype('float32')
            # batch_y[i,:,:,2] = Y_data_echo3_FA1.astype('float32')

        yield (batch_x, batch_y)
def multi_channel_image_generator_mat(directory, patch_size_FE,patch_size_PE, scale, batch_size, shuffle=True,
                                          seed=None):

    image_shape = (patch_size_FE, patch_size_PE, 6)
    y_image_shape = (patch_size_FE * scale, patch_size_PE * scale, 6)

    # s1 = sio.loadmat(directory+'/low_CompI_echo1.mat')
    # s1_list = list(s1.keys())
    # X_data = s1[s1_list[-1]]
    # s2 = sio.loadmat(directory+'/CompI_echo1.mat')
    # s2_list = list(s2.keys())
    # Y_data = s2[s2_list[-1]]


    #     nb_images = X_data_echo1_FA1.shape[0]
    # all echo in one mat
    # s1 = h5py.File(directory + '/low_CompI_final.mat')
    # X_data1 = s1['low_CompI_final'].value
    # X_data = X_data1[:,:,:,:]
    #
    # s1 = h5py.File(directory + '/CompI_final.mat')
    # Y_data1 = s1['CompI_final'].value
    # Y_data = Y_data1[:,:,:,:]
    # nb_images = Y_data.shape[1]
    # s1 = h5py.File(directory + '/low_CompI_final.mat')
    # X_data1 = s1['low_CompI_final'].value
    # X_data1 = np.transpose(X_data1, [3, 2, 1, 0])###nFE,nPE,nSL,nCH
    # X_data = X_data1[:, :, :, :]
    #
    # s1 = h5py.File(directory + '/CompI_final.mat')
    # Y_data1 = s1['CompI_final'].value
    # Y_data1 = np.transpose(Y_data1, [3, 2, 1, 0])
    # Y_data = Y_data1[:, :, :, :]
    # nb_images = Y_data.shape[2]
    # nb_images = 1
### for nPE*nSL, nCh, nFE*n_Volunteer
    s1 = h5py.File(directory + '/low_CompI_final.mat')
    X_data1 = s1['low_CompI_final'].value
    X_data1 = np.transpose(X_data1, [3, 2, 1, 0])###nPE,nSL,nFE*9,nCH
    X_data = X_data1[:, :, :, :]

    s1 = h5py.File(directory + '/CompI_final.mat')
    Y_data1 = s1['CompI_final'].value
    Y_data1 = np.transpose(Y_data1, [3, 2, 1, 0])
    Y_data = Y_data1[:, :, :, :]
    nb_images = Y_data.shape[2]



    # print("Found %d images." % nb_images)

    index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + image_shape)

        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):

            # batch_x[i, :, :, 0] = X_data_echo1_FA1[j, :, :].astype('float32')
            # batch_x[i, :, :, 1] = X_data_echo2_FA1[j, :, :].astype('float32')
            # batch_x[i, :, :, 2] = X_data_echo3_FA1[j, :, :].astype('float32')
            # batch_x[i, :, :, 3] = X_data_echo1_FA2[j, :, :].astype('float32')
            # batch_x[i, :, :, 4] = X_data_echo2_FA2[j, :, :].astype('float32')
            # batch_x[i, :, :, 5] = X_data_echo3_FA2[j, :, :].astype('float32')
            # batch_x[i, :, :, 0] = X_data[0,j, :, :].astype('float32')
            # batch_x[i, :, :, 1] = X_data[1,j, :, :].astype('float32')
            # batch_x[i, :, :, 2] = X_data[2,j, :, :].astype('float32')
            # batch_x[i, :, :, 3] = X_data[3,j, :, :].astype('float32')
            # batch_x[i, :, :, 4] = X_data[4,j, :, :].astype('float32')
            # batch_x[i, :, :, 5] = X_data[5,j, :, :].astype('float32')
            batch_y[i, :, :, 0] = Y_data[:,:,j,0].astype('float32')
            batch_y[i, :, :, 1] = Y_data[:,:,j,1].astype('float32')
            batch_y[i, :, :, 2] = Y_data[:,:,j,2].astype('float32')
            batch_y[i, :, :, 3] = Y_data[:,:,j,3].astype('float32')
            batch_y[i, :, :, 4] = Y_data[:,:,j,4].astype('float32')
            batch_y[i, :, :, 5] = Y_data[:,:,j,5].astype('float32')

            batch_x[i, :, :, 0] = X_data[:,:,j,0].astype('float32')
            batch_x[i, :, :, 1] = X_data[:,:,j,1].astype('float32')
            batch_x[i, :, :, 2] = X_data[:,:,j,2].astype('float32')
            batch_x[i, :, :, 3] = X_data[:,:,j,3].astype('float32')
            batch_x[i, :, :, 4] = X_data[:,:,j,4].astype('float32')
            batch_x[i, :, :, 5] = X_data[:,:,j,5].astype('float32')
            # batch_y[i, :, :, 0] = Y_data[0,j, :, :].astype('float32')
            # batch_y[i, :, :, 1] = Y_data[1,j, :, :].astype('float32')
            # batch_y[i, :, :, 2] = Y_data[2,j, :, :].astype('float32')
            # batch_y[i, :, :, 3] = Y_data[3,j, :, :].astype('float32')
            # batch_y[i, :, :, 4] = Y_data[4,j, :, :].astype('float32')
            # batch_y[i, :, :, 5] = Y_data[5,j, :, :].astype('float32')
            # batch_x[i,:,:,0] = X_data_echo1_FA1.astype('float32')
            # batch_x[i,:,:,1] = X_data_echo2_FA1.astype('float32')
            # batch_x[i,:,:,2] = X_data_echo3_FA1.astype('float32')
            # batch_y[i,:,:,0] = Y_data_echo1_FA1.astype('float32')
            # batch_y[i,:,:,1] = Y_data_echo2_FA1.astype('float32')
            # batch_y[i,:,:,2] = Y_data_echo3_FA1.astype('float32')

        yield (batch_x, batch_y)
def multi_channel_image_generator_complex_mat(directory,  patch_size_FE,patch_size_PE,scale, batch_size, shuffle=True, seed=None):

    image_shape = (patch_size_FE, patch_size_PE,12)
    y_image_shape = (patch_size_FE, patch_size_PE,12)


    # s1 = sio.loadmat(directory+'/low_CompI_echo1.mat')
    # s1_list = list(s1.keys())
    # X_data = s1[s1_list[-1]]
    # s2 = sio.loadmat(directory+'/CompI_echo1.mat')
    # s2_list = list(s2.keys())
    # Y_data = s2[s2_list[-1]]
    # low_FA1_echo1

    s1 = h5py.File(directory + '/low_CompI_final.mat')
    X_data1 = s1['low_CompI_final'].value
    X_data1 = np.transpose(X_data1, [3, 2, 1, 0])
    X_data = X_data1[:, :, :, :]

    s1 = h5py.File(directory + '/CompI_final.mat')
    Y_data1 = s1['CompI_final'].value
    Y_data1 = np.transpose(Y_data1, [3, 2, 1, 0])
    Y_data = Y_data1[:, :, :, :]
    nb_images = Y_data.shape[2]




    #print("Found %d images." % nb_images)

    index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + image_shape)

        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):

            #
            batch_y[i, :, :, 0] = Y_data[:,:,j,0].astype('float32')
            batch_y[i, :, :, 1] = Y_data[:,:,j,1].astype('float32')
            batch_y[i, :, :, 2] = Y_data[:,:,j,2].astype('float32')
            batch_y[i, :, :, 3] = Y_data[:,:,j,3].astype('float32')
            batch_y[i, :, :, 4] = Y_data[:,:,j,4].astype('float32')
            batch_y[i, :, :, 5] = Y_data[:,:,j,5].astype('float32')
            batch_y[i, :, :, 6] = Y_data[:,:,j,6].astype('float32')
            batch_y[i, :, :, 7] = Y_data[:,:,j,7].astype('float32')
            batch_y[i, :, :, 8] = Y_data[:,:,j,8].astype('float32')
            batch_y[i, :, :, 9] = Y_data[:,:,j,9].astype('float32')
            batch_y[i, :, :, 10] = Y_data[:,:,j,10].astype('float32')
            batch_y[i, :, :, 11] = Y_data[:,:,j,11].astype('float32')

            batch_x[i, :, :, 0] = X_data[:,:,j,0].astype('float32')
            batch_x[i, :, :, 1] = X_data[:,:,j,1].astype('float32')
            batch_x[i, :, :, 2] = X_data[:,:,j,2].astype('float32')
            batch_x[i, :, :, 3] = X_data[:,:,j,3].astype('float32')
            batch_x[i, :, :, 4] = X_data[:,:,j,4].astype('float32')
            batch_x[i, :, :, 5] = X_data[:,:,j,5].astype('float32')
            batch_x[i, :, :, 6] = X_data[:,:,j,6].astype('float32')
            batch_x[i, :, :, 7] = X_data[:,:,j,7].astype('float32')
            batch_x[i, :, :, 8] = X_data[:,:,j,8].astype('float32')
            batch_x[i, :, :, 9] = X_data[:,:,j,9].astype('float32')
            batch_x[i, :, :, 10] = X_data[:,:,j,10].astype('float32')
            batch_x[i, :, :, 11] = X_data[:,:,j,11].astype('float32')



        yield (batch_x, batch_y)
def multi_channel_image_generator_complex_mat_2(directory,  patch_size_PE,patch_size_FE,scale, batch_size, shuffle=True, seed=None):

    image_shape = (patch_size_PE, patch_size_FE,12)
    y_image_shape = (patch_size_PE, patch_size_FE,12)


    # s1 = sio.loadmat(directory+'/low_CompI_echo1.mat')
    # s1_list = list(s1.keys())
    # X_data = s1[s1_list[-1]]
    # s2 = sio.loadmat(directory+'/CompI_echo1.mat')
    # s2_list = list(s2.keys())
    # Y_data = s2[s2_list[-1]]
    # low_FA1_echo1

    s1 = h5py.File(directory + '/low_CompI_final.mat')
    X_data1 = s1['low_CompI_final'].value
    X_data = X_data1[:, :, :, :]

    s1 = h5py.File(directory + '/CompI_final.mat')
    Y_data1 = s1['CompI_final'].value
    Y_data = Y_data1[:, :, :, :]
    nb_images = Y_data.shape[1]




    #print("Found %d images." % nb_images)

    index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + image_shape)

        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):

            #
            batch_y[i, :, :, 0] = Y_data[0,j,:,:].astype('float32')
            batch_y[i, :, :, 1] = Y_data[1,j,:,:].astype('float32')
            batch_y[i, :, :, 2] = Y_data[2,j,:,:].astype('float32')
            batch_y[i, :, :, 3] = Y_data[3,j,:,:].astype('float32')
            batch_y[i, :, :, 4] = Y_data[4,j,:,:].astype('float32')
            batch_y[i, :, :, 5] = Y_data[5,j,:,:].astype('float32')
            batch_y[i, :, :, 6] = Y_data[6,j,:,:].astype('float32')
            batch_y[i, :, :, 7] = Y_data[7,j,:,:].astype('float32')
            batch_y[i, :, :, 8] = Y_data[8,j,:,:].astype('float32')
            batch_y[i, :, :, 9] = Y_data[9,j,:,:].astype('float32')
            batch_y[i, :, :, 10] = Y_data[10,j,:,:].astype('float32')
            batch_y[i, :, :, 11] = Y_data[11,j,:,:].astype('float32')

            batch_x[i,:,:,0] = X_data[0,j,:,:].astype('float32')
            batch_x[i,:,:,1] = X_data[1,j,:,:].astype('float32')
            batch_x[i,:,:,2] = X_data[2,j,:,:].astype('float32')
            batch_x[i,:,:,3] = X_data[3,j,:,:].astype('float32')
            batch_x[i,:,:,4] = X_data[4,j,:,:].astype('float32')
            batch_x[i,:,:,5] = X_data[5,j,:,:].astype('float32')
            batch_x[i,:,:,6] = X_data[6,j,:,:].astype('float32')
            batch_x[i,:,:,7] = X_data[7,j,:,:].astype('float32')
            batch_x[i,:,:,8] = X_data[8,j,:,:].astype('float32')
            batch_x[i,:,:,9] = X_data[9,j,:,:].astype('float32')
            batch_x[i,:,:,10] = X_data[10,j,:,:].astype('float32')
            batch_x[i,:,:,11] = X_data[11,j,:,:].astype('float32')



        yield (batch_x, batch_y)
def image_generator_complex_mat(directory, patch_size,scale, batch_size, shuffle=True, seed=None):

    image_shape = (patch_size, patch_size,2)
    y_image_shape = (patch_size*scale, patch_size*scale,2)



    s1 = h5py.File(directory+'/low_CompI_real.mat')
    X_data_reala = s1['low_CompI_real'].value
    X_data_real = X_data_reala[14,:,:]
    s1 = h5py.File(directory+'/low_CompI_imag.mat')
    X_data_imaga = s1['low_CompI_imag'].value
    X_data_imag = X_data_imaga[14,:,:]

    s2 = h5py.File(directory+'/CompI_real.mat')
    Y_data_reala = s2['CompI_real'].value
    Y_data_real = Y_data_reala[14,:,:]

    s2 = h5py.File(directory+'/CompI_imag.mat')
    Y_data_imaga= s2['CompI_imag'].value
    Y_data_imag = Y_data_imaga[14,:,:]

    nb_images = X_data_real.shape[0]
    # s1 = h5py.File(directory+'/low_CompI_echo1_FA1_real.mat')
    # X_data_real = s1['low_CompI_echo1_FA1_real'].value
    # s1 = h5py.File(directory+'/low_CompI_echo1_FA1_imag.mat')
    # X_data_imag = s1['low_CompI_echo1_FA1_imag'].value
    #
    # s2 = h5py.File(directory+'/CompI_echo1_FA1_real.mat')
    # Y_data_real = s2['CompI_echo1_FA1_real'].value
    # s2 = h5py.File(directory+'/CompI_echo1_FA1_imag.mat')
    # Y_data_imag = s2['CompI_echo1_FA1_imag'].value
    # nb_images = X_data_real.shape[0]


    #print("Found %d images." % nb_images)

    index_generator = _index_generator(nb_images, batch_size, shuffle, seed)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + image_shape)

        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):
            # batch_x[i, :, :, 0] = X_data_real[j,:,:]
            #
            # batch_x[i,:,:,1] = X_data_imag[j,:,:]
            # batch_x[i,:,:,:] = batch_x[i,:,:,:].astype('float32')
            #
            # batch_y[i, :, :, 0] = Y_data_real[j,:,:]
            # batch_y[i, :, :, 1] = Y_data_imag[j,:,:]
            # batch_y[i, :, :, :] = batch_y[i,:,:,:].astype('float32')
            batch_x[i, :, :, 0] = X_data_real

            batch_x[i,:,:,1] = X_data_imag
            batch_x[i,:,:,:] = batch_x[i,:,:,:].astype('float32')

            batch_y[i, :, :, 0] = Y_data_real
            batch_y[i, :, :, 1] = Y_data_imag
            batch_y[i, :, :, :] = batch_y[i,:,:,:].astype('float32')


        mean_y = np.mean(batch_y)
        image_target = batch_y - mean_y


        yield (batch_x, image_target)
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
def multi_channel_image_generator_tfrecord(directory, patch_size, batch_size, shuffle=True, seed=None):
    image_shape = (patch_size, patch_size, NUM_CHENNELS)
    y_image_shape = (patch_size, patch_size, NUM_CHENNELS)
    X_data_echo1_real_FA1, X_data_echo1_imag_FA1, Y_data_echo1_real_FA1, Y_data_echo1_imag_FA1 =tfrecord_read(Filenames=directory,batch_size= batch_size, shuffle=True)
    X_data = tf.concat([X_data_echo1_real_FA1, X_data_echo1_imag_FA1], axis = 3)
    Y_data = tf.concat([Y_data_echo1_real_FA1, Y_data_echo1_imag_FA1], axis = 3)

    return  X_data, Y_data
def tfrecord_read(Filenames, batch_size, crop_patch_size,shuffle=True):
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
    low = tf.reshape(low, [crop_patch_size, crop_patch_size])

    high = tf.decode_raw(features['CompI'], tf.float64)
    high = tf.reshape(high, [crop_patch_size, crop_patch_size])
    low_batch, high_batch = tf.train.shuffle_batch([low, high], batch_size, capacity=50000, min_after_dequeue=20000)
    low_image = tf.reshape(low_batch,[batch_size,crop_patch_size,crop_patch_size,1])
    high_image = tf.reshape(high_batch,[batch_size,crop_patch_size,crop_patch_size,1])


    return low_image,high_image
def tfrecord_read_6echo(Filenames, batch_size, crop_patch_FE, crop_patch_PE, Num_CHANNELS,shuffle=True):
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
    low = tf.reshape(low, [crop_patch_FE, crop_patch_PE,Num_CHANNELS])

    high = tf.decode_raw(features['CompI'], tf.float64)
    high = tf.reshape(high, [crop_patch_FE, crop_patch_PE,Num_CHANNELS])

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
    low_image = tf.reshape(low_batch,[batch_size,crop_patch_FE, crop_patch_PE,Num_CHANNELS])
    high_image = tf.reshape(high_batch,[batch_size,crop_patch_FE, crop_patch_PE,Num_CHANNELS])


    return low_image,high_image
def tfrecord_test_read(Filenames, batch_size, crop_patch_size,shuffle=True):
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
    low = tf.reshape(low, [crop_patch_size, crop_patch_size])

    high = tf.decode_raw(features['CompI'], tf.float64)
    high = tf.reshape(high, [crop_patch_size, crop_patch_size])
    low_batch, high_batch = tf.train.batch([low, high], batch_size)
    low_image = tf.reshape(low_batch,[batch_size,crop_patch_size,crop_patch_size,1])
    high_image = tf.reshape(high_batch,[batch_size,crop_patch_size,crop_patch_size,1])


    return low_image,high_image
def distort_imgs(data,img_size):
    """ data augumentation """
    train_x , train_y = data
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = train_x.reshape((-1,img_size,img_size,1))
    train_y = train_y.reshape((-1,img_size,img_size,1))
    train_x, train_y = tl.prepro.flip_axis_multi([train_x, train_y],axis=1, is_random=True)  # left right
    train_x, train_y= tl.prepro.elastic_transform_multi([train_x, train_y], alpha=720, sigma=24, is_random=True)
    train_x, train_y= tl.prepro.rotation_multi([train_x, train_y], rg=20, is_random=True, fill_mode='constant')  # nearest, constant
    train_x, train_y = tl.prepro.shift_multi([train_x, train_y], wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
    train_x, train_y = tl.prepro.shear_multi([train_x, train_y], 0.05,  is_random=True, fill_mode='constant')
    train_x, train_y = tl.prepro.zoom_multi([train_x, train_y], zoom_range=[0.9, 1.1], is_random=True,  fill_mode='constant')
    return train_x, train_y
# if __name__ == "__main__":
#     # Transform the images once, then run the main code to scale images
#
#     # Change scaling factor to increase the scaling factor
#
#     # Set true_upscale to True to generate smaller training images that will then be true upscaled.
#     # Leave as false to create same size input and output images
#
#     transform_images(input_path_low, input_path_high, output_path_low, output_path_high, 512, 512, 512, 0, max_nb_images=-1)
#     pass



