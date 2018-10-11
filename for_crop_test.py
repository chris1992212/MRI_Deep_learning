'''
2-D Convolutional Neural Networks using TensorFlow library for MR Image reconstruction.

'''
import numpy as np
from scipy.misc import imsave, imread, imresize
import tensorflow as tf
import SRCNN_models
from data_inputs import *
from SRCNN_configs import *
os.environ['CUDA_VISIBLE_DEVICES'] = ' 0'

def main():
    # Parameters
    BATCH_SIZE = 1
    crop_size = 48
    NUM_TESTING_STEPS = 48*50
    # Create Input and Output
    low_res_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, crop_size, crop_size, NUM_CHENNELS])
    high_res_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, crop_size, crop_size, NUM_CHENNELS])

    # CNN model
    low_res_image = tf.reshape(low_res_holder, shape=[-1, crop_size, crop_size, NUM_CHENNELS])
    high_res_image = tf.reshape(high_res_holder, shape=[-1, crop_size, crop_size, NUM_CHENNELS])
    inferences = SRCNN_models.create_model(MODEL_NAME, low_res_image,is_train= True)

    testing_loss = SRCNN_models.loss(inferences, high_res_image, name='testing_loss', weights_decay=0)
    srcing_loss = SRCNN_models.loss(low_res_image, high_res_image, name='src_loss', weights_decay=0)

    '''
    TensorFlow Session
    '''
    # start TensorFlow session
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(init)
    saver.restore(sess, 'model2/mymodel')
    mse =  np.zeros((NUM_TESTING_STEPS,1))
    mse2 =  np.zeros((NUM_TESTING_STEPS,1))
    msed = np.zeros((NUM_TESTING_STEPS,1))
    # train_epoch = multi_channel_image_generator_complex_mat("train\\ceshi\\test_Amp_3echo_FA2_for_every_picture/", patch_size_low, 1,
    #                                                         BATCH_SIZE, False,
    #                                                         seed=None)
    # train_epoch = multi_channel_image_generator_complex_mat("train\\ceshi\\test_FA2_echo2_echo_norm/", patch_size_low, 1,
    #                                                         BATCH_SIZE, False,
    #                                                         seed=None)
    # train_epoch = image_generator_mat("train\\test_Amp_all_echo_in_one_channel/", patch_size_low, 1, BATCH_SIZE, False,
    #                                   seed=None)
    # 生成队列
    # batch_low, batch_high = tfrecord_read("Amp_all_echo_one_channel.tfrecord", BATCH_SIZE, crop_size)
    batch_low, batch_high = tfrecord_read_6echo("Amp_6channel.tfrecord", BATCH_SIZE, crop_size)

    out = np.zeros((crop_size,crop_size,NUM_TESTING_STEPS))
    low = np.zeros((crop_size,crop_size,NUM_TESTING_STEPS))
    high = np.zeros((crop_size,crop_size,NUM_TESTING_STEPS))
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)
    for  i in range(NUM_TESTING_STEPS):
        batch_xs, batch_ys = sess.run([batch_low, batch_high])
        # tf.summary.image('input', tf.uint8(batch_xs), 10)
        recon, high_res_images, low_res_images = sess.run([inferences, high_res_image, low_res_image],
                                                          feed_dict={low_res_holder: batch_xs,
                                                                     high_res_holder: batch_ys})
        mse2[i] = sess.run(srcing_loss, feed_dict={low_res_holder: batch_xs, high_res_holder: batch_ys})
        mse[i] = sess.run(testing_loss, feed_dict={low_res_holder: batch_xs, high_res_holder: batch_ys})
        # out[:,:,i*3:3*(i+1)] = recon[0,:,:,:]
        # low[:,:,i*3:3*(i+1)] = low_res_images[0,:,:,:]
        # high[:,:,i*3:3*(i+1)] = high_res_images[0,:,:,:]
        # save for single_channel
        # out[:,:,i:(i+1)] = recon[0,:,:,:]
        # low[:,:,i:(i+1)] = low_res_images[0,:,:,:]
        # high[:,:,i:(i+1)] = high_res_images[0,:,:,:]
        # mse= 0
        # mse2[i] = src_loss
        # mse3[i] = ssiming_loss1
        # # mse4[i] = ssiming_loss2
        # recon_FA1_echo1 = recon[:, :, :, 0]
        # recon_FA1_echo2 = recon[:, :, :, 1]
        # recon_FA1_echo3 = recon[:, :, :, 2]
        # result_FA1_echo1 = recon_FA1_echo1.reshape(patch_size, patch_size).astype(np.float32) * 255
        # result_FA1_echo2 = recon_FA1_echo2.reshape(patch_size, patch_size).astype(np.float32) * 255
        # result_FA1_echo3 = recon_FA1_echo3.reshape(patch_size, patch_size).astype(np.float32) * 255
        # # # mse= 0
        # # # mse2[i] = src_loss
        # # # mse3[i] = ssiming_loss1
        # # # # mse4[i] = ssiming_loss2
        # recon_FA1_echo1 = recon[:,:,:,0:2]
        # # recon_FA1_echo2 = recon[:,:,:,2:4]
        # # recon_FA1_echo3 = recon[:,:,:,4:6]
        # # recon_FA2_echo1 = recon[:,:,:,6:8]
        # # recon_FA2_echo2 = recon[:,:,:,8:10]
        # # recon_FA2_echo3 = recon[:,:,:,10:12]
        # #
        # result_FA1_echo1 = np.sqrt(np.sum(np.square(recon_FA1_echo1),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # result_FA1_echo2 = np.sqrt(np.sum(np.square(recon_FA1_echo2),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # result_FA1_echo3 = np.sqrt(np.sum(np.square(recon_FA1_echo3),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # # result_FA2_echo1 = np.sqrt(np.sum(np.square(recon_FA2_echo1),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # # result_FA2_echo2 = np.sqrt(np.sum(np.square(recon_FA2_echo2),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # # result_FA2_echo3 = np.sqrt(np.sum(np.square(recon_FA2_echo3),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # #
        # imsave('result2/final_FA1_echo1_'+str(i)+'.bmp', result_FA1_echo1)
        # imsave('result2/final_FA1_echo2_'+str(i)+'.bmp', result_FA1_echo2)
        # imsave('result2/final_FA1_echo3_'+str(i)+'.bmp',result_FA1_echo3)
        # # imsave('data/result2/final_FA1_echo1_'+str(i)+'.bmp', result_FA1_echo1)
        # # imsave('data/result2/final_FA1_echo2_'+str(i)+'.bmp', result_FA1_echo2)
        # # imsave('data/result2/final_FA1_echo3_'+str(i)+'.bmp',result_FA1_echo3)
        # # imsave('data/result/final_FA2_echo1_'+str(i)+'.bmp', result_FA2_echo1)
        # # imsave('data/result/final_FA2_echo2_'+str(i)+'.bmp', result_FA2_echo2)
        # # imsave('data/result/final_FA2_echo3_'+str(i)+'.bmp', result_FA2_echo3)
        # #
        # #
        # recon_FA1_echo1 = low_res_images[:,:,:,0]
        # recon_FA1_echo2 = low_res_images[:,:,:,1]
        # recon_FA1_echo3 = low_res_images[:,:,:,2]
        # # recon_FA1_echo1 = low_res_images[:,:,:,0:2]
        # # recon_FA1_echo2 = low_res_images[:,:,:,2:4]
        # # recon_FA1_echo3 = low_res_images[:,:,:,4:6]
        # # recon_FA2_echo1 = low_res_images[:,:,:,6:8]
        # # recon_FA2_echo2 = low_res_images[:,:,:,8:10]
        # # recon_FA2_echo3 = low_res_images[:,:,:,10:12]
        # #
        # result_FA1_echo1 = recon_FA1_echo1.reshape(patch_size, patch_size).astype(np.float32) * 255
        # result_FA1_echo2 = recon_FA1_echo2.reshape(patch_size, patch_size).astype(np.float32) * 255
        # result_FA1_echo3 = recon_FA1_echo3.reshape(patch_size, patch_size).astype(np.float32) * 255
        # # recon_FA1_echo1 = low_res_images[:,:,:,0:2]
        # # recon_FA1_echo2 = low_res_images[:,:,:,2:4]
        # # recon_FA1_echo3 = low_res_images[:,:,:,4:6]
        # # recon_FA2_echo1 = low_res_images[:,:,:,6:8]
        # # recon_FA2_echo2 = low_res_images[:,:,:,8:10]
        # # recon_FA2_echo3 = low_res_images[:,:,:,10:12]
        # #
        # result_FA1_echo1 = np.sqrt(np.sum(np.square(recon_FA1_echo1),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # result_FA1_echo2 = np.sqrt(np.sum(np.square(recon_FA1_echo2),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # result_FA1_echo3 = np.sqrt(np.sum(np.square(recon_FA1_echo3),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # # result_FA2_echo1 = np.sqrt(np.sum(np.square(recon_FA2_echo1),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # # result_FA2_echo2 = np.sqrt(np.sum(np.square(recon_FA2_echo2),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # # result_FA2_echo3 = np.sqrt(np.sum(np.square(recon_FA2_echo3),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # # #
        # imsave('result2/low_FA1_echo1_' + str(i) + '.bmp', result_FA1_echo1)
        # imsave('result2/low_FA1_echo2_' + str(i) + '.bmp', result_FA1_echo2)
        # imsave('result2/low_FA1_echo3_' + str(i) + '.bmp', result_FA1_echo3)
        # # imsave('data/result2/low_FA1_echo1_'+str(i)+'.bmp', result_FA1_echo1)
        # # imsave('data/result2/low_FA1_echo2_'+str(i)+'.bmp', result_FA1_echo2)
        # # imsave('data/result2/low_FA1_echo3_'+str(i)+'.bmp',result_FA1_echo3)
        # # imsave('data/result/low_FA2_echo1_' + str(i) + '.bmp', result_FA2_echo1)
        # # imsave('data/result/low_FA2_echo2_' + str(i) + '.bmp', result_FA2_echo2)
        # # imsave('data/result/low_FA2_echo3_' + str(i) + '.bmp', result_FA2_echo3)
        # #
        # recon_FA1_echo1 = high_res_images[:, :, :, 0]
        # recon_FA1_echo2 = high_res_images[:, :, :, 1]
        # recon_FA1_echo3 = high_res_images[:, :, :, 2]
        # # recon_FA1_echo1 = high_res_images[:,:,:,:]
        # # recon_FA1_echo2 = high_res_images[:,:,:,2:4]
        # # recon_FA1_echo3 = high_res_images[:,:,:,4:6]
        # # recon_FA2_echo1 = high_res_images[:,:,:,6:8]
        # # recon_FA2_echo2 = high_res_images[:,:,:,8:10]
        # # recon_FA2_echo3 = high_res_images[:,:,:,10:12]
        # result_FA1_echo1 = recon_FA1_echo1.reshape(patch_size, patch_size).astype(np.float32) * 255
        # result_FA1_echo2 = recon_FA1_echo2.reshape(patch_size, patch_size).astype(np.float32) * 255
        # result_FA1_echo3 = recon_FA1_echo3.reshape(patch_size, patch_size).astype(np.float32) * 255
        # # # recon_FA1_echo1 = high_res_images[:,:,:,0:2]
        # # recon_FA1_echo2 = high_res_images[:,:,:,2:4]
        # # recon_FA1_echo3 = high_res_images[:,:,:,4:6]
        # # recon_FA2_echo1 = high_res_images[:,:,:,6:8]
        # # recon_FA2_echo2 = high_res_images[:,:,:,8:10]
        # # recon_FA2_echo3 = high_res_images[:,:,:,10:12]
        # # result_FA1_echo1 = np.sqrt(np.sum(np.square(recon_FA1_echo1),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # # result_FA1_echo2 = np.sqrt(np.sum(np.square(recon_FA1_echo2),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # # result_FA1_echo3 = np.sqrt(np.sum(np.square(recon_FA1_echo3),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # # result_FA2_echo1 = np.sqrt(np.sum(np.square(recon_FA2_echo1),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # # result_FA2_echo2 = np.sqrt(np.sum(np.square(recon_FA2_echo2),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # # result_FA2_echo3 = np.sqrt(np.sum(np.square(recon_FA2_echo3),axis =3)).reshape(patch_size, patch_size).astype(np.float32) * 255
        # #
        # imsave('result2/high_FA1_echo1_' + str(i) + '.bmp', result_FA1_echo1)
        # imsave('result2/high_FA1_echo2_' + str(i) + '.bmp', result_FA1_echo2)
        # imsave('result2/high_FA1_echo3_' + str(i) + '.bmp', result_FA1_echo3)
        # imsave('data/result2/high_FA1_echo1_'+str(i)+'.bmp', result_FA1_echo1)
        # imsave('data/result2/high_FA1_echo2_'+str(i)+'.bmp', result_FA1_echo2)
        # imsave('data/result2/high_FA1_echo3_'+str(i)+'.bmp',result_FA1_echo3)
        # imsave('data/result/high_FA2_echo1_' + str(i) + '.bmp', result_FA2_echo1)
        # imsave('data/result/high_FA2_echo2_' + str(i) + '.bmp', result_FA2_echo2)
        # imsave('data/result/high_FA2_echo3_' + str(i) + '.bmp', result_FA2_echo3)
        print('i:%d, test_MSE: %.7f, src_MSE: %.7f' % (i,mse[i], mse2[i]))
    # save for single_channel
    # sio.savemat('out.mat', {'out': out})
    # sio.savemat('low.mat', {'low': low})
    # sio.savemat('high.mat', {'high': high})
    # mse /= NUM_TESTING_STEPS
    # mse2 /= 192
    print('i: %d ,min_test_MSE: %.7f, max_test_MSE: %.7f, ave_test_MSE: %.7f, min_src_MSE: %.7f, max_src_MSE: %.7f, ave_src_MSE: %.7f' % (i, mse.min(),mse.max(),mse.mean(), mse2.min(),mse2.max(),mse2.mean()))

if __name__ == '__main__':
    main()



