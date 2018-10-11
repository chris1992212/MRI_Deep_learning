'''
2-D Convolutional Neural Networks using TensorFlow library for MR Image reconstruction.

'''
import numpy as np
from scipy.misc import imsave, imread, imresize
from tensorlayer.prepro import threading_data
import SRCNN_models
from data_inputs import *
from SRCNN_configs import *
os.environ['CUDA_VISIBLE_DEVICES'] = ' 1'
import utils
#### basci config
LEARNING_RATE = config.LEARNING_RATE
TRAINING_EPOCHS = config.TRAINING_EPOCHS
BATCH_SIZE = config.BATCH_SIZE
DISPLAY_STEP = config.DISPLAY_STEP
crop_size_PE = config.crop_size_PE
crop_size_FE = config.crop_size_FE
validation_PE = config.validation_PE
validation_FE = config.validation_FE
Train_filename = config.Train_filename
Test_filename = config.Test_filename
Test_filename2 = config.Test_filename2
early_stop_number = config.early_stop_number
NUM_CHANNELS = config.NUM_CHANNELS
MODEL_NAME = config.MODEL_NAME
save_model_filename_best = config.save_model_filename_best
save_model_filename = config.save_model_filename
restore_model_filename = config.restore_model_filename
def main():
    # Parameters
    BATCH_SIZE = 1




    patch_size_PE = validation_PE
    patch_size_FE = validation_FE
    NUM_TESTING_STEPS = np.int(48/BATCH_SIZE)
    # Create Input and Output
    low_res_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, patch_size_FE, patch_size_PE, NUM_CHANNELS])
    high_res_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, patch_size_FE, patch_size_PE, NUM_CHANNELS])

    # CNN model
    low_res_image = tf.reshape(low_res_holder, shape=[-1, patch_size_FE, patch_size_PE, NUM_CHANNELS])
    high_res_image = tf.reshape(high_res_holder, shape=[-1, patch_size_FE, patch_size_PE, NUM_CHANNELS])

    inferences = SRCNN_models.create_model(MODEL_NAME, low_res_image,n_out= NUM_CHANNELS,is_train= False)
    # inferences = tf.abs(inferences)

    testing_loss = SRCNN_models.loss(inferences, high_res_image, name='testing_loss', weights_decay=0)
    srcing_loss = SRCNN_models.loss(low_res_image, high_res_image, name='src_loss', weights_decay=0)


    '''
    TensorFlow Session
    '''
    # start TensorFlow session
    init = tf.initialize_all_variables()
    saver = tf.train.Saver(tf.global_variables())
    sess = tf.InteractiveSession()
    sess.run(init)
    saver.restore(sess, restore_model_filename)
    # saver.restore(sess, 'model_Amp_6channel_final_BN/mymodel')

    mse =  np.zeros((NUM_TESTING_STEPS,1))
    mse2 =  np.zeros((NUM_TESTING_STEPS,1))
    # test_ssim_loss = np.zeros((NUM_TESTING_STEPS,1))
    # src_ssim_loss = np.zeros((NUM_TESTING_STEPS,1))
    # train_epoch = multi_channel_image_generator_complex_mat("train\\ceshi\\test_Amp_3echo_FA2_for_every_picture/", patch_size_low, 1,
    #                                                         BATCH_SIZE, False,
    #                                                         seed=None)
    test_epoch = multi_channel_image_generator_mat(Test_filename, patch_size_FE,patch_size_PE, 1, BATCH_SIZE, False,
                                   seed=None)
    # test_epoch = image_generator_mat("train\\test_real_data", patch_size_FE,patch_size_PE, 1, BATCH_SIZE, False,
    #                                seed=None)
    # train_epoch = multi_channel_image_generator_complex_mat("train\\ceshi\\caiyue", patch_size_FE,patch_size_PE, 1, BATCH_SIZE, False,
    #                                seed=None)

    # train_epoch = multi_channel_image_generator_complex_mat("train\\Abdomen/", patch_size_PE,patch_size_FE, 1, BATCH_SIZE, False,
                                   # seed=None)
    # train_epoch = image_generator_mat("train\\ceshi\\Abdomen", patch_size_PE, patch_size_FE, 1, BATCH_SIZE, False,
    #                                   seed=None)
    # train_epoch = multi_channel_image_generator_complex_mat("train\\ceshi", patch_size_PE, patch_size_FE,
    #                                                         1, BATCH_SIZE, False,
    #                                                         seed=None)
    # train_epoch = image_generator_mat("train\\test_Amp_all_echo_in_one_channel/", patch_size_low, 1, BATCH_SIZE, False,
    #                                   seed=None)
    out = np.zeros((patch_size_FE,patch_size_PE,NUM_TESTING_STEPS,NUM_CHANNELS))
    low = np.zeros((patch_size_FE,patch_size_PE,NUM_TESTING_STEPS,NUM_CHANNELS))
    high = np.zeros((patch_size_FE,patch_size_PE,NUM_TESTING_STEPS,NUM_CHANNELS))
    for  i in range(NUM_TESTING_STEPS):

        batch_xs, batch_ys = next(test_epoch)

        # tf.summary.image('input', tf.uint8(batch_xs), 10)
        recon, high_res_images, low_res_images = sess.run([inferences, high_res_image, low_res_image],
                                                          feed_dict={low_res_holder: batch_xs,
                                                                     high_res_holder: batch_ys})
        # print("clock7:%s" % (time.clock()))

        # test_ssim_loss[i] = threading_data([_ for _ in zip(high_res_images, recon)], fn=utils.ssim)
        # src_ssim_loss[i] = threading_data([_ for _ in zip(high_res_images, low_res_images)], fn=utils.ssim)
        mse[i],mse2[i]= sess.run([testing_loss,srcing_loss], feed_dict={low_res_holder: batch_xs, high_res_holder: batch_ys})
        out[:,:,i,:] = recon[0,:,:,:].reshape(patch_size_FE,patch_size_PE,NUM_CHANNELS)
        low[:,:,i,:] = low_res_images[0,:,:,:].reshape(patch_size_FE,patch_size_PE,NUM_CHANNELS)
        high[:,:,i,:] = high_res_images[0,:,:,:].reshape(patch_size_FE,patch_size_PE,NUM_CHANNELS)

        print('i:%d, test_MSE: %.7f, src_MSE: %.7f' % (i,mse[i], mse2[i]))
        # print('i:%d, ssim_test: %.7f, ssim_src: %.7f',i, test_ssim_loss[i],src_ssim_loss[i])

    # save for single_channel
    saving_path = config.saving_path
    tl.files.exists_or_mkdir(saving_path)
    sio.savemat(os.path.join(saving_path,'out.mat'), {'out': out})
    sio.savemat('low.mat', {'low': low})
    sio.savemat('high.mat', {'high': high})
    # # mse /= NUM_TESTING_STEPS
    # mse2 /= 192
    print('i: %d ,min_test_MSE: %.7f, max_test_MSE: %.7f, ave_test_MSE: %.7f, min_src_MSE: %.7f, max_src_MSE: %.7f, ave_src_MSE: %.7f' % (i, mse.min(),mse.max(),mse.mean(), mse2.min(),mse2.max(),mse2.mean()))
    # print('ssim_test: %.7f, ssim_src: %.7f',test_ssim_loss.mean(),src_ssim_loss.mean())
    # end = time.clock()
    # print ("clock5:%s" % (end))


if __name__ == '__main__':
    main()



