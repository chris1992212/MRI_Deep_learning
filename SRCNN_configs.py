from easydict import EasyDict as edict
import json
import os
# data path and log path
config = edict()
config.testing = edict()

# model
config.MODEL_NAME = 'u_net_bn_new_2'          # srcnn, vgg7, vgg_deconv_7,SRresnet,u_net,EDSR
config.INPUT_SIZE = 384                       # the image size input to the network
config.PE_size_ori = 288
config.FE_size_ori = 384
config.Scale =1
config.LEARNING_RATE = 0.0001
config.TRAINING_EPOCHS = 250000
config.BATCH_SIZE = 2
config.DISPLAY_STEP = 10
config.crop_size_PE = 288
config.crop_size_FE = 384
config.validation_PE = 288
config.validation_FE = 384
config.early_stop_number = 40
config.NUM_CHANNELS = 6
config.training_BN = True

# testing
config.testing.patch_size_PE = 288
config.testing.patch_size_FE = 384
config.testing_BN = False

#### Filename

config.Train_filename = os.path.join('train')
config.Test_filename = os.path.join('train','test_real_data','xujun','xujun_with_DPA_simulation')
config.Test_filename2 = os.path.join('train','6_channel_hanlu')
config.tfrecord_filename = os.path.join('Amp_6channel.tfrecord')
config.restore_model_filename = os.path.join('model_Amp_6channel_bn_7_12',config.MODEL_NAME,'good','mymodel')
config.save_model_filename = os.path.join('model_Amp_one_channel_bn_8_15_all_echo',config.MODEL_NAME,'common','mymodel')
config.save_model_filename_best = os.path.join('model_Amp_one_channel_bn_8_15_all_echo',config.MODEL_NAME,'good','mymodel')
# testing for all pictures
config.Test_Batch_size = 4
config.TESTING_NUM = int(288/config.Test_Batch_size)
config.log_dir = "log_{}".format(config.MODEL_NAME)
config.saving_path = os.path.join('F:\matlab\Data_address_cml\BrainQuant_AI\Data','xujun','xujun_with_DPA_simulation')

###
def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")


