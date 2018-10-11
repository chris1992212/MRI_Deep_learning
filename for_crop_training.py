'''
2-D Convolutional Neural Networks using TensorFlow library for MR Image reconstruction.
Author: MingliangChen
'''
from tqdm import tqdm
from data_inputs import *
import SRCNN_models
import tensorflow as tf
import os
from tensorlayer.prepro import threading_data
import utils
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Parameters
from SRCNN_configs import config

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
# training_BN = config.training_BN
testing_BN = config.testing_BN

early_stop_number = config.early_stop_number
NUM_CHANNELS = config.NUM_CHANNELS
MODEL_NAME = config.MODEL_NAME
tfrecord_filename = config.tfrecord_filename
TESTING_NUM = config.TESTING_NUM
Test_Batch_size = config.Test_Batch_size
### filename
tfrecord_filename = config.tfrecord_filename
save_model_filename = config.save_model_filename
save_model_filename_best = config.save_model_filename_best
restore_model_filename = config.restore_model_filename
tl.files.exists_or_mkdir(save_model_filename)
tl.files.exists_or_mkdir(save_model_filename_best)
# log
log_dir = "log_{}".format(MODEL_NAME)
tl.files.exists_or_mkdir(log_dir)
log_all, log_eval, log_all_filename, log_eval_filename = utils.logging_setup(log_dir)
log_config(log_all_filename, config)
log_config(log_eval_filename, config)

# Create Input and Output
with tf.name_scope('input'):
    low_res_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, crop_size_FE, crop_size_PE, NUM_CHANNELS], name = 'low')
    high_res_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, crop_size_FE, crop_size_PE, NUM_CHANNELS])
    low_res_holder_validation = tf.placeholder(tf.float32, shape=[Test_Batch_size, validation_FE, validation_PE, NUM_CHANNELS])
    high_res_holder_validation = tf.placeholder(tf.float32, shape=[Test_Batch_size, validation_FE, validation_PE, NUM_CHANNELS])



# CNN model
inferences = SRCNN_models.create_model(MODEL_NAME, low_res_holder,n_out= NUM_CHANNELS, is_train= True,reuse= False)
inferences_validation = SRCNN_models.create_model(MODEL_NAME, low_res_holder_validation,n_out= NUM_CHANNELS,is_train= False,reuse = True)

# Loss function (MSE)
# training_loss = SRCNN_models.loss(inferences, high_res_image, name='training_loss', weights_decay=0)
training_loss = SRCNN_models.loss(inferences, high_res_holder, name='training_loss', weights_decay=0)
# validating_loss = SRCNN_models.loss(inferences_validation, high_res_holder_validation, name='validating_loss', weights_decay=0)
validation_loss = SRCNN_models.loss(inferences_validation, high_res_holder_validation, name='validation_loss', weights_decay=0)
srcing_loss = SRCNN_models.loss(low_res_holder_validation, high_res_holder_validation, name='src_loss', weights_decay=0)

training_accuracy = SRCNN_models.PSNRLoss(inferences, high_res_holder, name= 'training_accurancy')
# ssim_loss1, ssim_loss2  = SRCNN_models.ssim_loss(inferences, high_res_image,BATCH_SIZE, PATCH_SIZE=patch_size, name='ssim_loss')


global_step = tf.Variable(0, trainable=False, name='global_step')
# learning_rate = tf.train.piecewise_constant(
#     global_step,
#     [2000, 5000, 8000, 12000, 16000],
#     [0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
# )
# learning_rate = tf.train.inverse_time_decay(0.0001, global_step, 10000, 0.5,name = 'learning_rate')
learning_rate = 0.0001
# with tf.variable_scope('learning_rate'):
#     lr_v = tf.Variable(0.00001, trainable=False)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(training_loss, global_step=global_step)

'''
TensorFlow Session
'''
# start TensorFlow session
init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())
tf.summary.scalar('training_loss', training_loss)
tf.summary.scalar('training_accuracy', training_accuracy)
# tf.summary.scalar('learning_rate', lr_v)
tf.summary.scalar('Learning_rate', learning_rate)
# tf.summary.scalar('Global_step_var', global_step)

# tf.summary.scalar('validat_loss', validating_loss)
merged_summary_op = tf.summary.merge_all()
##### 目的：不恢复新的学习率；
# graph_restore = tf.get_default_graph() #此时默认图就是导入的图
# graph_restore.clear_collection('learning_rate') #删除以前的集合，假如finetuning后用新的代替原来的
# # graph_restore.clear_collection('global_step_var')
# tf.add_to_collection('learning_rate', learning_rate)  # 重新添加到新集合
# tf.add_to_collection('global_step_var', global_step)  # 重新添加到新集合

#生成队列
# batch_low,batch_high = tfrecord_read("Amp_all_echo_one_channel.tfrecord",BATCH_SIZE,crop_size)
# batch_low,batch_high = tfrecord_read_6echo("Amp_6channel.tfrecord",BATCH_SIZE,crop_size)
batch_low,batch_high = tfrecord_read_6echo(tfrecord_filename,BATCH_SIZE,crop_size_FE,crop_size_PE,NUM_CHANNELS)


sess = tf.InteractiveSession()
#logs_path = "graph"
summary_writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)
saver.restore(sess,restore_model_filename)

# visualisation variables
train_accuracies = []
epochs_completed = 0
index_in_epoch = 0
# train_epoch = multi_channel_image_generator_complex_mat("data\\train/ceshi/multi_channel_after_scale",patch_size_low, Scale, BATCH_SIZE, True,
#                                              seed=None)
# train_epoch = image_generator_complex_mat("data\\train\\zqq_data", patch_size_low, Scale,BATCH_SIZE, True,
#                                              seed=None)
# train_epoch = image_generator_mat(Train_filename,crop_size_PE, crop_size_FE,1,BATCH_SIZE, True,
#                                              seed=None)
train_epoch = multi_channel_image_generator_mat(Train_filename,crop_size_FE, crop_size_PE,1,BATCH_SIZE, True,
                                             seed=None)
# test_epoch = image_generator_mat(Test_filename,validation_FE, validation_PE,1,Test_Batch_size, False,
#                                              seed=None)
# test_epoch2 = image_generator_mat(Test_filename2,validation_FE, validation_PE,1,Test_Batch_size, False,
#                                              seed=None)
test_epoch = multi_channel_image_generator_mat(Test_filename,validation_FE, validation_PE,1,Test_Batch_size, False,
                                             seed=None)
test_epoch2 = multi_channel_image_generator_mat(Test_filename2,validation_FE, validation_PE,1,Test_Batch_size, False,
                                             seed=None)
# train_validation = image_generator_mat("data\\train\\ceshi\\test_Amp_FA1_echo1/", patch_size_low,1,BATCH_SIZE, True,
#                                              seed=None)
# train_epoch = image_generator("data\\train\\ceshi/new_1_8/", patch_size_low,BATCH_SIZE, True,
#                                              seed=None)

coord = tf.train.Coordinator()
tf.train.start_queue_runners(sess=sess,coord=coord)

best_mse = 1
ear_stop = 40
best_epoch = 1
best_mse = np.array(best_mse,dtype= np.float64)
best_validation =1

for epoch in tqdm(range(TRAINING_EPOCHS)):
    # get new batch
    # batch_xs, batch_ys = sess.run([batch_low, batch_high])

    start = time.clock()
    batch_xs, batch_ys  = next(train_epoch)
    _, summary = sess.run([train_step, merged_summary_op],
                           feed_dict={low_res_holder: batch_xs, high_res_holder: batch_ys})
    out = sess.run(inferences,feed_dict={low_res_holder: batch_xs, high_res_holder: batch_ys})
    summary_writer.add_summary(summary,  epoch)
    end = time.clock()
    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if epoch%100 ==0:
        # batch_xs_validation, batch_ys_validation = next(train_validation)
        # batch_xs,batch_ys = sess.run([batch_low,batch_high])
        train_accuracy, train_lost = sess.run([training_accuracy, training_loss],
                                              feed_dict={low_res_holder: batch_xs, high_res_holder: batch_ys})
        # validation_lost = sess.run(validating_loss,
        #                                       feed_dict={low_res_holder_validation: batch_xs_validation, high_res_holder_validation: batch_ys_validation})
        train_accuracies.append(train_accuracy)
        # print('epochs %d training_cost => %.7f train_accuracy => %.7f validation_lost => %.7f' % (i, train_lost, train_accuracy, validation_lost))
        print('epochs %d training_cost => %.7f train_accuracy => %.7f ' % (epoch, train_lost, train_accuracy))

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    # if i % DISPLAY_STEP == 0 or (i + 1) == TRAINING_EPOCHS:
    #     # train_accuracy, train_lost = sess.run([training_accuracy, training_loss],
    #     #                                       feed_dict={low_res_holder: batch_xs, high_res_holder: batch_ys})
    #     # print('epochs %d training_cost => %.7f train_accuracy => %.7f' % (i, train_lost, train_accuracy))
    #     # train_accuracies.append(train_accuracy)
        save_path = saver.save(sess, save_model_filename)
########### Testing Part
        if epoch % 1000 == 0:
            saver.restore(sess, save_model_filename)
            test_ssim_loss = np.zeros((TESTING_NUM, 1))
            src_ssim_loss = np.zeros((TESTING_NUM, 1))
            mse = np.zeros((TESTING_NUM, 1))
            mse2 = np.zeros((TESTING_NUM, 1))
            for j in range(TESTING_NUM):
                batch_xs_validation, batch_ys_validation = next(test_epoch)
                # tf.summary.image('input', tf.uint8(batch_xs), 10)
                recon, high_res_images, low_res_images = sess.run([inferences_validation, high_res_holder_validation, low_res_holder_validation],
                                                              feed_dict={low_res_holder_validation: batch_xs_validation,
                                                                         high_res_holder_validation: batch_ys_validation})
                # recon = np.abs(recon)
                # test_ssim_loss[j] = threading_data([_ for _ in zip(high_res_images, recon)], fn=utils.ssim)
                # src_ssim_loss[j] = threading_data([_ for _ in zip(high_res_images, low_res_images)], fn=utils.ssim)
                mse[j], mse2[j] = sess.run([validation_loss, srcing_loss],
                                       feed_dict={low_res_holder_validation: batch_xs_validation, high_res_holder_validation: batch_ys_validation})
            print(  'i: %d ,ave_test_MSE: %.7f,ave_src_MSE: %.7f' % ( epoch, mse.mean(), mse2.mean()))
            # print('ssim_test: %.7f, ssim_src: %.7f' % (test_ssim_loss.mean(), src_ssim_loss.mean()))
            l = mse.mean()
            if l < best_mse:
                save_path = saver.save(sess, save_model_filename_best)
                best_mse = mse.mean()
                ear_stop = early_stop_number
                best_epoch = epoch
            else:
                ear_stop -=1
            print('best_epoch: %d, ear_stop: %d' %(best_epoch, ear_stop))
            if ear_stop==0:
                print('best_mse: %.7f' %(best_mse))
                break
            # log = "Epoch: {}\n SSIM test val: {:8}, SSIM src val: {:8}, mse val: {:8}".format(
            #     epoch + 1,
            #     test_ssim_loss.mean(),
            #     src_ssim_loss.mean(),
            #     mse.mean())

            mse_2 = np.zeros((TESTING_NUM, 1))
            mse2_2 = np.zeros((TESTING_NUM, 1))
            for j in range(TESTING_NUM):
                batch_xs_validation, batch_ys_validation = next(test_epoch2)
                # tf.summary.image('input', tf.uint8(batch_xs), 10)
                recon, high_res_images, low_res_images = sess.run([inferences_validation, high_res_holder_validation, low_res_holder_validation],
                                                              feed_dict={low_res_holder_validation: batch_xs_validation,
                                                                         high_res_holder_validation: batch_ys_validation})
                # recon = np.abs(recon)
                # test_ssim_loss[j] = threading_data([_ for _ in zip(high_res_images, recon)], fn=utils.ssim)
                # src_ssim_loss[j] = threading_data([_ for _ in zip(high_res_images, low_res_images)], fn=utils.ssim)
                mse_2[j], mse2_2[j] = sess.run([validation_loss, srcing_loss],
                                       feed_dict={low_res_holder_validation: batch_xs_validation, high_res_holder_validation: batch_ys_validation})
            print(  'i: %d ,ave_test_MSE: %.7f,ave_src_MSE: %.7f' % ( epoch, mse_2.mean(), mse2_2.mean()))

            log = "Best_epoch: {}\n Epoch: {}\n mse val: {:8}\n mse_src: {:8}\n mse_training:{:8} \n mse_train_src:{:8} ".format(best_epoch,
                epoch + 1,
                mse.mean(),
                mse2.mean(),
                mse_2.mean(),
                mse2_2.mean())
            print(log)
            log_all.debug(log)
            log_eval.info(log)
    #     if i % (DISPLAY_STEP * 10) == 0 and i:
    #         DISPLAY_STEP *= 10
    #     if train_lost< best_train_lost:
    #         best_train_lost = train_lost
    #         save_path = saver.save(sess, 'model1\mymodel')
    #     if validation_lost < best_validation:
    #         best_validation = validation_lost
    #         save_path = saver.save(sess, 'model3\mymodel')
    #         last_improvement = i
    # if i - last_improvement > require_improvement:
    #     print("No improvement found in a while, stopping optimization.")
    #     break
    # train on batch

    # print("training time for once: ", str(end))
coord.request_stop()  # queue需要关闭，否则报错

save_path = saver.save(sess, 'model5\mymodel')
print("Model saved in file:", save_path)
sess.close()
