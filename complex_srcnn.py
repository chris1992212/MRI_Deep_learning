'''
2-D Convolutional Neural Networks using TensorFlow library for MR Image reconstruction.
Author: MingliangChen
'''
from tqdm import tqdm
from data_inputs import *
from SRCNN_configs import *
import SRCNN_models
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Parameters
from SRCNN_configs import *
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 50000
BATCH_SIZE = BATCH_SIZE_TRAINING
DISPLAY_STEP = 10
patch_size_PE =288

# Create Input and Output
with tf.name_scope('input'):
    low_res_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, patch_size_PE, patch_size_low, NUM_CHENNELS])
    high_res_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, patch_size_PE, patch_size, NUM_CHENNELS])
    low_res_holder_validation = tf.placeholder(tf.float32, shape=[BATCH_SIZE, patch_size_PE, patch_size_low, NUM_CHENNELS])
    high_res_holder_validation = tf.placeholder(tf.float32, shape=[BATCH_SIZE, patch_size_PE, patch_size, NUM_CHENNELS])
    lr = tf.placeholder('float')
    # tf.summary.image("input_image", tf.cast(low_res_holder, tf.uint8))
    # tf.summary.image("taget_image", tf.cast(high_res_holder, tf.uint8))
    # tf.summary.image("input_image", low_res_holder,10)
    # tf.summary.image("taget_image", high_res_holder,10)
# with tf.name_scope('input_reshape'):
#     low_res_image = tf.reshape(low_res_holder, shape=[BATCH_SIZE, patch_size/2, patch_size/2, NUM_CHENNELS])
#     high_res_image = tf.reshape( high_res_holder, shape=[BATCH_SIZE, patch_size, patch_size, NUM_CHENNELS])



# CNN model
inferences = SRCNN_models.create_model(MODEL_NAME, low_res_holder, is_train= True)
# inferences_validation = SRCNN_models.create_model(MODEL_NAME, low_res_holder_validation,is_train= True)

# Loss function (MSE)
# training_loss = SRCNN_models.loss(inferences, high_res_image, name='training_loss', weights_decay=0)
training_loss = SRCNN_models.loss(inferences, high_res_holder, name='training_loss', weights_decay=0)
# validating_loss = SRCNN_models.loss(inferences_validation, high_res_holder_validation, name='validating_loss', weights_decay=0)

training_accuracy = SRCNN_models.PSNRLoss(inferences, high_res_holder, name= 'training_accurancy')
# ssim_loss1, ssim_loss2  = SRCNN_models.ssim_loss(inferences, high_res_image,BATCH_SIZE, PATCH_SIZE=patch_size, name='ssim_loss')


global_step = tf.Variable(0, trainable=False, name='global_step')
# learning_rate = tf.train.piecewise_constant(
#     global_step,
#     [2000, 5000, 8000, 12000, 16000],
#     [0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
# )
# learning_rate = tf.train.inverse_time_decay(0.0001, global_step, 10000, 2)
train_step = tf.train.AdamOptimizer(0.0001).minimize(training_loss, global_step=global_step)

'''
TensorFlow Session
'''
# start TensorFlow session
init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())
tf.summary.scalar('training_loss', training_loss)
tf.summary.scalar('training_accuracy', training_accuracy)
# tf.summary.scalar('Learning_rate', learning_rate)
# tf.summary.scalar('validat_loss', validating_loss)

merged_summary_op = tf.summary.merge_all()
sess = tf.InteractiveSession()
#logs_path = "graph"
summary_writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)
saver.restore(sess, 'model2/mymodel')

# visualisation variables
train_accuracies = []
epochs_completed = 0
index_in_epoch = 0
# num_examples = train_images.shape[0]
# train_epoch = multi_channel_image_generator_complex_mat("data\\train/ceshi/multi_channel_after_scale",patch_size_low, Scale, BATCH_SIZE, True,
#                                              seed=None)
# train_epoch = image_generator_complex_mat("data\\train\\zqq_data", patch_size_low, Scale,BATCH_SIZE, True,
#                                              seed=None)
# train_epoch = image_generator_mat("data\\train\\ceshi\\Amp_FA1_echo1/", patch_size_low,1,BATCH_SIZE, True,
#                                              seed=None)
# train_validation = image_generator_mat("data\\train\\ceshi\\test_Amp_FA1_echo1/", patch_size_low,1,BATCH_SIZE, True,
#                                              seed=None)
# train_epoch = image_generator("data\\train\\ceshi/new_1_8/", patch_size_low,BATCH_SIZE, True,
#                                              seed=None)
# train_validation = image_generator("data\\train\\ceshi/new_1_8/", patch_size_low,BATCH_SIZE, True,
#                                              seed=None)
# train_epoch = image_generator_complex_mat("data\\train\\ceshi/single_echo_mat", patch_size_low,1,BATCH_SIZE, True,
#                                               seed=None)
# train_validation = multi_channel_image_generator_complex_mat("data\\train/ceshi/test_multi_channel_after_scale",patch_size_low, Scale, BATCH_SIZE, True,
#                                              seed=None)
# train_epoch = multi_channel_image_generator_complex_mat("data\\train\\ceshi\\FA1_3echo_picture_norm/", patch_size_low, 1, BATCH_SIZE, True,
#                                   seed=None)
# train_validation = multi_channel_image_generator_complex_mat("data\\train\\ceshi\\test_FA1_3echo_picture_norm/", patch_size_low, 1, BATCH_SIZE, True,
#                                   seed=None)
train_epoch = multi_channel_image_generator_mat("train\\Amp_6echo/factor_4_sense", patch_size_PE,patch_size_low, 1, BATCH_SIZE, True,
                                  seed=None)
# train_validation = multi_channel_image_generator_complex_mat("train\\Amp_6echo/", patch_size_low, 1, BATCH_SIZE, True,
#                                   seed=None)
# train_epoch = image_generator_mat("data\\train\\stage/", patch_size_low, Scale, BATCH_SIZE, False,
#                                   seed=None)
# train_validation = image_generator_mat("train\\stage/", patch_size_low, 1, BATCH_SIZE, False,
#                                   seed=None)
best_train_lost = 1
best_validation =1
require_improvement =100000
for i in tqdm(range(TRAINING_EPOCHS)):
    # get new batch
    # batch_xs, batch_ys = next_batch(BATCH_SIZE)

    start = time.clock()
    batch_xs, batch_ys  = next(train_epoch)
    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%100 ==0:
        # batch_xs_validation, batch_ys_validation = next(train_validation)
        train_accuracy, train_lost = sess.run([training_accuracy, training_loss],
                                              feed_dict={low_res_holder: batch_xs, high_res_holder: batch_ys})
        # validation_lost = sess.run(validating_loss,
        #                                       feed_dict={low_res_holder_validation: batch_xs_validation, high_res_holder_validation: batch_ys_validation})
        train_accuracies.append(train_accuracy)
        # print('epochs %d training_cost => %.7f train_accuracy => %.7f validation_lost => %.7f' % (i, train_lost, train_accuracy, validation_lost))
        print('epochs %d training_cost => %.7f train_accuracy => %.7f ' % (i, train_lost, train_accuracy))

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    # if i % DISPLAY_STEP == 0 or (i + 1) == TRAINING_EPOCHS:
    #     # train_accuracy, train_lost = sess.run([training_accuracy, training_loss],
    #     #                                       feed_dict={low_res_holder: batch_xs, high_res_holder: batch_ys})
    #     # print('epochs %d training_cost => %.7f train_accuracy => %.7f' % (i, train_lost, train_accuracy))
    #     # train_accuracies.append(train_accuracy)
        save_path = saver.save(sess, 'model2\mymodel')
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
    _, summary = sess.run([train_step, merged_summary_op],
                           feed_dict={low_res_holder: batch_xs, high_res_holder: batch_ys})
    out = sess.run(inferences,feed_dict={low_res_holder: batch_xs, high_res_holder: batch_ys})
    summary_writer.add_summary(summary,  i)
    end = time.clock()
    # print("training time for once: ", str(end))

save_path = saver.save(sess, 'model\mymodel')
print("Model saved in file:", save_path)
sess.close()
