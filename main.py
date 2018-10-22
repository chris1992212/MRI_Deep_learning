import tensorflow as tf
import os
from model import BrainQuantAI_Part_one

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("image_size", 80, "The size of images after cropping")
flags.DEFINE_integer("label_size", 80, "The size of label")
flags.DEFINE_integer("test_FE",384,"The size in FE direction of testing images")
flags.DEFINE_integer("test_PE",288,"The size in PE direction of testing images")
flags.DEFINE_integer("c_dim", 6, "The size of channel")

flags.DEFINE_integer("epoch", 15000, "Number of epoch")
flags.DEFINE_integer("Batch_Size",16,"Batch Size of Training Data")
flags.DEFINE_integer("TESTING_NUM",48,"Number of Testing Data")

#---------------------------Filenames-----------------------#
flags.DEFINE_string("tfrecord_train","Amp_6channel.tfrecord","training Data saved as tfrecord")

save_model_filename = os.path.join('Model','mymodel')
flags.DEFINE_string("save_model_filename",save_model_filename,"saved network model")
test_filename = os.path.join('Data','test','data3','without_PF_net')
flags.DEFINE_string("testing_filename",test_filename,"Testing data file")
restore_model_filename = os.path.join('Good_model_for_Amp','model_Amp_6channel_bn_7_12','u_net_bn_new_2','good','mymodel')
flags.DEFINE_string("restore_model_filename",restore_model_filename,"restore model data file")




flags.DEFINE_boolean("is_train", True, "if the train")
flags.DEFINE_integer("scale", 2, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 17, "the size of stride")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory")
flags.DEFINE_float("learning_rate", 1e-5, "The learning rate")
flags.DEFINE_integer("batch_size", 32, "the size of batch")
flags.DEFINE_integer("des_block_H", 8, "the size dense_block layer number")
flags.DEFINE_integer("des_block_ALL", 8,"the size dense_block")
flags.DEFINE_string("result_dir", "result", "Name of result directory")
flags.DEFINE_integer("growth_rate", 16, "the size of growrate")
flags.DEFINE_string("test_img", "", "test_img")

def main(_): #?
    with tf.Session() as sess:
        brainquant = BrainQuantAI_Part_one(sess,
                      image_size = FLAGS.image_size,
                      label_size = FLAGS.label_size,
                      is_train = FLAGS.is_train,
                      batch_size = FLAGS.batch_size,
                      c_dim = FLAGS.c_dim,
                      test_FE = FLAGS.test_FE,
                      test_PE=FLAGS.test_PE
                         )

        # brainquant.train(FLAGS)
        brainquant.pred_test(FLAGS)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ' 0'

    tf.app.run()