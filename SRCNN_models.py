import tensorflow.contrib.slim as slim
from SRCNN_layers import *
from SRCNN_configs import *
import tensorlayer as tl
from tensorlayer.layers import *
# from loss_func import  *
from EDSR_utils import *
import utils
def create_model(name, patches, n_out, is_train=False,reuse = False):
    if name == 'srcnn':
        return srcnn_935(patches,reuse= reuse)
    elif name == 'vgg7':
        return vgg7(patches)
    elif name =='vgg_deconv_7':
        return vgg_deconv_7(patches)
    elif name == 'u_net':
        return u_net(patches,n_out= n_out,reuse = reuse)
    elif name == 'u_net_new':
        return u_net_new(patches,n_out= n_out,reuse= reuse)
    elif name == 'u_net_bn':
        return u_net_bn(patches,n_out= n_out,is_train= is_train,reuse= reuse)
    elif name == 'u_net_bn_new':
        return u_net_bn_new(patches, n_out= n_out, is_train= is_train,reuse= reuse)
    elif name == 'u_net_bn_new_2':
        return u_net_bn_new_2(patches, n_out= n_out, is_train= is_train,reuse= reuse)
    elif name =='srcnn_9751':
        return srcnn_9751(patches,is_train)
    elif name == 'SRresnet':
        return SRresnet(patches, n_out= n_out, is_train= is_train,reuse= reuse)
    elif name == 'EDSR':
        return EDSR(patches)
    elif name == 'SRDENSE':
        return SRDENSE(patches, n_out= n_out, is_train= is_train,reuse= reuse)

def u_net(x,  n_out=12,reuse= False):  # Do I need to change n_out here ???
    _, nx, ny, nz = x.get_shape().as_list()
    # mean_x = tf.reduce_mean(x)
    # x = x - mean_x
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    with tf.variable_scope("u_net",reuse = reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl.layers.InputLayer(x, name='inputs')
        # inputs = tf.layers.batch_normalization(inputs =inputs.outputs, axis= -1, training = True)
        # inputs = tl.layers.InputLayer(inputs, name ='input_batch_layer')

        conv1 = tl.layers.Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
        pool1 = tl.layers.MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')

        net1 = conv1.outputs
        #variable_summaries(net1, 'net_1')
        # tf.summary.image('net_1',net1[:,:,:,1],10)

        conv2 = tl.layers.Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
        pool2 = tl.layers.MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')

        net2 = conv2.outputs
        #variable_summaries(net2, 'net_2')
        # tf.summary.image('net_2',net2,10)

        conv3 = tl.layers.Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
        pool3 = tl.layers.MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')

        net3 = conv3.outputs
        # variable_summaries(net3, 'net_3')
        #tf.summary.image('net_3',net3,10)

        conv4 = tl.layers.Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')
        pool4 = tl.layers.MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')

        net4 = conv4.outputs
        # variable_summaries(net4, 'net_4')
        #tf.summary.image('net_4',net4,10)

        conv5 = tl.layers.Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
        conv5 = tl.layers.Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')

        net5 = conv5.outputs
        # variable_summaries(net5, 'net_5')
        # tf.summary.image('net_5',net5,10)

        print(" * After conv: %s" % conv5.outputs)

        up4 = tl.layers.DeConv2d(conv5, 512, (3, 3), out_size=(nx / 8, ny / 8), strides=(2, 2),
                       padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = tl.layers.ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
        conv4 = tl.layers.Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_2')

        netup4 = conv4.outputs
        # variable_summaries(netup4, 'netup_4')
        # tf.summary.image('netup_4',netup4,10)

        up3 = tl.layers.DeConv2d(conv4, 256, (3, 3), out_size=(nx / 4, ny / 4), strides=(2, 2),
                       padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = tl.layers.ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
        conv3 = tl.layers.Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_2')

        netup3 = conv3.outputs
        # variable_summaries(netup3, 'netup_3')
        # tf.summary.image('netup_3',netup3,10)

        up2 = tl.layers.DeConv2d(conv3, 128, (3, 3), out_size=(nx / 2, ny / 2), strides=(2, 2),
                       padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = tl.layers.ConcatLayer([up2, conv2], concat_dim=3, name='concat2')
        conv2 = tl.layers.Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_2')

        netup2 = conv2.outputs
        # variable_summaries(netup2, 'netup_2')
        # tf.summary.image('netup_2',netup2,10)

        up1 = tl.layers.DeConv2d(conv2, 64, (3, 3), out_size=(nx / 1, ny / 1), strides=(2, 2),
                       padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = tl.layers.ConcatLayer([up1, conv1], concat_dim=3, name='concat1')
        conv1 = tl.layers.Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_2')

        netup1 = conv1.outputs
        # variable_summaries(netup1, 'netup_1')
        # tf.summary.image('netup_1',netup1,10)

        conv1 = tl.layers.Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')
        # tf.summary.image('final', tf.cast(conv1.outputs,tf.uint8), 1)
        # tf.summary.image('final', conv1.outputs, 10)
        conv1 = tf.add(conv1.outputs, inputs.outputs)

        # print(" * Output: %s" % conv1.outputs)
        #outputs = tl.act.pixel_wise_softmax(conv1.outputs)
        return conv1

def u_net_bn(x, n_out=12,is_train = False, reuse=False):  # Do I need to change n_out here ???
    _, nx, ny, nz = x.get_shape().as_list()
    # mean_x = tf.reduce_mean(x)
    # x = x - mean_x
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1, 0.02)

    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl.layers.InputLayer(x, name='inputs')
        # inputs = tf.layers.batch_normalization(inputs =inputs.outputs, axis= -1, training = True)
        # inputs = tl.layers.InputLayer(inputs, name ='input_batch_layer')

        conv1 = tl.layers.Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv1_2')
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn1')
        pool1 = tl.layers.MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')

        net1 = conv1.outputs
        # variable_summaries(net1, 'net_1')
        # tf.summary.image('net_1',net1[:,:,:,1],10)

        conv2 = tl.layers.Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv2_2')
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn2')
        pool2 = tl.layers.MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')

        net2 = conv2.outputs
        # variable_summaries(net2, 'net_2')
        # tf.summary.image('net_2',net2,10)

        conv3 = tl.layers.Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv3_2')
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn3')
        pool3 = tl.layers.MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')

        net3 = conv3.outputs
        # variable_summaries(net3, 'net_3')
        # tf.summary.image('net_3',net3,10)

        conv4 = tl.layers.Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv4_2')
        conv4 = BatchNormLayer(conv4, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn4')
        pool4 = tl.layers.MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')

        net4 = conv4.outputs
        # variable_summaries(net4, 'net_4')
        # tf.summary.image('net_4',net4,10)

        conv5 = tl.layers.Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv5_1')
        conv5 = tl.layers.Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv5_2')
        conv5 = BatchNormLayer(conv5, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn5')
        net5 = conv5.outputs
        # variable_summaries(net5, 'net_5')
        # tf.summary.image('net_5',net5,10)

        print(" * After conv: %s" % conv5.outputs)

        up4 = tl.layers.DeConv2d(conv5, 512, (3, 3), out_size=(nx / 8, ny / 8), strides=(2, 2),
                                 padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = tl.layers.ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
        conv4 = tl.layers.Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv4_2')
        conv4 = BatchNormLayer(conv4, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn4')
        netup4 = conv4.outputs
        # variable_summaries(netup4, 'netup_4')
        # tf.summary.image('netup_4',netup4,10)

        up3 = tl.layers.DeConv2d(conv4, 256, (3, 3), out_size=(nx / 4, ny / 4), strides=(2, 2),
                                 padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = tl.layers.ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
        conv3 = tl.layers.Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv3_2')
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn3')
        netup3 = conv3.outputs
        # variable_summaries(netup3, 'netup_3')
        # tf.summary.image('netup_3',netup3,10)

        up2 = tl.layers.DeConv2d(conv3, 128, (3, 3), out_size=(nx / 2, ny / 2), strides=(2, 2),
                                 padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = tl.layers.ConcatLayer([up2, conv2], concat_dim=3, name='concat2')
        conv2 = tl.layers.Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv2_2')
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn2')
        netup2 = conv2.outputs
        # variable_summaries(netup2, 'netup_2')
        # tf.summary.image('netup_2',netup2,10)

        up1 = tl.layers.DeConv2d(conv2, 64, (3, 3), out_size=(nx / 1, ny / 1), strides=(2, 2),
                                 padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = tl.layers.ConcatLayer([up1, conv1], concat_dim=3, name='concat1')
        conv1 = tl.layers.Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv1_2')
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn1')
        netup1 = conv1.outputs
        # variable_summaries(netup1, 'netup_1')
        # tf.summary.image('netup_1',netup1,10)

        conv1 = tl.layers.Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')
        # tf.summary.image('final', tf.cast(conv1.outputs,tf.uint8), 1)
        # tf.summary.image('final', conv1.outputs, 10)
        conv1 = tf.add(conv1.outputs, inputs.outputs)

        # print(" * Output: %s" % conv1.outputs)
        # outputs = tl.act.pixel_wise_softmax(conv1.outputs)
        return conv1

def u_net_bn_new(x, n_out=12, is_train=False, reuse=False):  # Do I need to change n_out here ???
    _, nx, ny, nz = x.get_shape().as_list()
    # mean_x = tf.reduce_mean(x)
    # x = x - mean_x
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1, 0.02)

    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl.layers.InputLayer(x, name='inputs')
        # inputs = tf.layers.batch_normalization(inputs =inputs.outputs, axis= -1, training = True)
        # inputs = tl.layers.InputLayer(inputs, name ='input_batch_layer')

        conv1 = tl.layers.Conv2d(inputs, 64, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv1_1')
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv1_2')
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn1_2')
        pool1 = tl.layers.MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')

        net1 = conv1.outputs
        # variable_summaries(net1, 'net_1')
        # tf.summary.image('net_1',net1[:,:,:,1],10)

        conv2 = tl.layers.Conv2d(pool1, 128, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv2_1')
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv2_2')
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn2_2')
        pool2 = tl.layers.MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')

        net2 = conv2.outputs
        # variable_summaries(net2, 'net_2')
        # tf.summary.image('net_2',net2,10)

        conv3 = tl.layers.Conv2d(pool2, 256, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv3_1')
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv3_2')
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn3_2')
        pool3 = tl.layers.MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')

        net3 = conv3.outputs
        # variable_summaries(net3, 'net_3')
        # tf.summary.image('net_3',net3,10)

        conv4 = tl.layers.Conv2d(pool3, 512, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv4_1')
        conv4 = BatchNormLayer(conv4, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv4_2')
        conv4 = BatchNormLayer(conv4, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn4_2')
        pool4 = tl.layers.MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')

        net4 = conv4.outputs
        # variable_summaries(net4, 'net_4')
        # tf.summary.image('net_4',net4,10)

        conv5 = tl.layers.Conv2d(pool4, 1024, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv5_1')
        conv5 = BatchNormLayer(conv5, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn5_1')
        conv5 = tl.layers.Conv2d(conv5, 1024, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv5_2')
        conv5 = BatchNormLayer(conv5, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init,
                               name='bn5_2')
        net5 = conv5.outputs
        # variable_summaries(net5, 'net_5')
        # tf.summary.image('net_5',net5,10)

        print(" * After conv: %s" % conv5.outputs)

        up4 = tl.layers.DeConv2d(conv5, 512, (3, 3), out_size=(nx / 8, ny / 8), strides=(2, 2),
                                 padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = BatchNormLayer(up4, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                             name='ucov_bn4_1')
        up4 = tl.layers.ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
        conv4 = tl.layers.Conv2d(up4, 512, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv4_2')
        conv4 = BatchNormLayer(conv4, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn4_2')
        netup4 = conv4.outputs
        # variable_summaries(netup4, 'netup_4')
        # tf.summary.image('netup_4',netup4,10)

        up3 = tl.layers.DeConv2d(conv4, 256, (3, 3), out_size=(nx / 4, ny / 4), strides=(2, 2),
                                 padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = BatchNormLayer(up3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                             name='ucov_bn3_1')
        up3 = tl.layers.ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
        conv3 = tl.layers.Conv2d(up3, 256, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv3_2')
        conv3 = BatchNormLayer(conv3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn3_2')
        netup3 = conv3.outputs
        # variable_summaries(netup3, 'netup_3')
        # tf.summary.image('netup_3',netup3,10)

        up2 = tl.layers.DeConv2d(conv3, 128, (3, 3), out_size=(nx / 2, ny / 2), strides=(2, 2),
                                 padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = BatchNormLayer(up2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                             name='ucov_bn2_1')
        up2 = tl.layers.ConcatLayer([up2, conv2], concat_dim=3, name='concat2')
        conv2 = tl.layers.Conv2d(up2, 128, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv2_2')
        conv2 = BatchNormLayer(conv2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn2_2')
        netup2 = conv2.outputs
        # variable_summaries(netup2, 'netup_2')
        # tf.summary.image('netup_2',netup2,10)

        up1 = tl.layers.DeConv2d(conv2, 64, (3, 3), out_size=(nx / 1, ny / 1), strides=(2, 2),
                                 padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = BatchNormLayer(up1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                             name='ucov_bn1_1')
        up1 = tl.layers.ConcatLayer([up1, conv1], concat_dim=3, name='concat1')
        conv1 = tl.layers.Conv2d(up1, 64, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv1_2')
        conv1 = BatchNormLayer(conv1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn1_2')
        netup1 = conv1.outputs
        # variable_summaries(netup1, 'netup_1')
        # tf.summary.image('netup_1',netup1,10)

        # conv1 = tl.layers.Conv2d(conv1, n_out, (1, 1), act=tf.nn.tanh, name='uconv1')
        # tf.summary.image('final', tf.cast(conv1.outputs,tf.uint8), 1)
        conv1 = tl.layers.Conv2d(conv1, 32, (3, 3), act=tf.nn.relu, name='uconv1')
        conv1 = tl.layers.Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1_new')

        # tf.summary.image('final', conv1.outputs, 10)
        conv1 = tf.add(conv1.outputs, inputs.outputs)

        # print(" * Output: %s" % conv1.outputs)
        # outputs = tl.act.pixel_wise_softmax(conv1.outputs)
        return conv1

def u_net_bn_new_2(x, n_out=12, is_train=False, reuse=False):
    batch_size, nx, ny, nz = x.get_shape().as_list()
    # mean_x = tf.reduce_mean(x)
    # x = x - mean_x
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1, 0.02)

    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl.layers.InputLayer(x, name='inputs')
        # inputs = tf.layers.batch_normalization(inputs =inputs.outputs, axis= -1, training = True)
        # inputs = tl.layers.InputLayer(inputs, name ='input_batch_layer')

        conv1 = tl.layers.Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv1_2')
        conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init,
                               name='bn1')
        pool1 = tl.layers.MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')

        net1 = conv1.outputs
        # variable_summaries(net1, 'net_1')
        # tf.summary.image('net_1',net1[:,:,:,1],10)

        conv2 = tl.layers.Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv2_2')
        conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init,
                               name='bn2')
        pool2 = tl.layers.MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')

        net2 = conv2.outputs
        # variable_summaries(net2, 'net_2')
        # tf.summary.image('net_2',net2,10)

        conv3 = tl.layers.Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv3_2')
        conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init,
                               name='bn3')
        pool3 = tl.layers.MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')

        net3 = conv3.outputs
        # variable_summaries(net3, 'net_3')
        # tf.summary.image('net_3',net3,10)

        conv4 = tl.layers.Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv4_2')
        conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init,
                               name='bn4')
        pool4 = tl.layers.MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')

        net4 = conv4.outputs
        # variable_summaries(net4, 'net_4')
        # tf.summary.image('net_4',net4,10)

        conv5 = tl.layers.Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv5_1')
        conv5 = tl.layers.Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='conv5_2')
        conv5 = BatchNormLayer(conv5, is_train=is_train, gamma_init=gamma_init,
                               name='bn5')
        net5 = conv5.outputs
        # variable_summaries(net5, 'net_5')
        # tf.summary.image('net_5',net5,10)

        print(" * After conv: %s" % conv5.outputs)

        up4 = tl.layers.DeConv2d(conv5, 512, (3, 3), out_size=[tf.to_int32(tf.shape(x)[1] / 8),tf.to_int32(tf.shape(x)[2]/ 8)], strides=(2, 2),
                                 padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = BatchNormLayer(up4, is_train=is_train, gamma_init=gamma_init,
                             name='ucov_bn4_1')
        up4 = tl.layers.ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
        conv4 = tl.layers.Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv4_2')
        conv4 = BatchNormLayer(conv4, is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn4_2')
        netup4 = conv4.outputs
        # variable_summaries(netup4, 'netup_4')
        # tf.summary.image('netup_4',netup4,10)

        up3 = tl.layers.DeConv2d(conv4, 256, (3, 3), out_size=[tf.to_int32(tf.shape(x)[1] / 4),tf.to_int32(tf.shape(x)[2]/ 4)], strides=(2, 2),
                                 padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = BatchNormLayer(up3, is_train=is_train, gamma_init=gamma_init,
                             name='ucov_bn3_1')
        up3 = tl.layers.ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
        conv3 = tl.layers.Conv2d(up3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv3_2')
        conv3 = BatchNormLayer(conv3, is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn3_2')
        netup3 = conv3.outputs
        # variable_summaries(netup3, 'netup_3')
        # tf.summary.image('netup_3',netup3,10)

        up2 = tl.layers.DeConv2d(conv3, 128, (3, 3), out_size=[tf.to_int32(tf.shape(x)[1] / 2),tf.to_int32(tf.shape(x)[2]/ 2)], strides=(2, 2),
                                 padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = BatchNormLayer(up2, is_train=is_train, gamma_init=gamma_init,
                             name='ucov_bn2_1')
        up2 = tl.layers.ConcatLayer([up2, conv2], concat_dim=3, name='concat2')
        conv2 = tl.layers.Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv2_2')
        conv2 = BatchNormLayer(conv2, is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn2_2')
        netup2 = conv2.outputs
        # variable_summaries(netup2, 'netup_2')
        # tf.summary.image('netup_2',netup2,10)

        up1 = tl.layers.DeConv2d(conv2, 64, (3, 3), out_size=[tf.to_int32(tf.shape(x)[1] ),tf.to_int32(tf.shape(x)[2])], strides=(2, 2),
                                 padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = BatchNormLayer(up1, is_train=is_train, gamma_init=gamma_init,
                             name='ucov_bn1_1')
        up1 = tl.layers.ConcatLayer([up1, conv1], concat_dim=3, name='concat1')
        conv1 = tl.layers.Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                 name='uconv1_2')
        conv1 = BatchNormLayer(conv1, is_train=is_train, gamma_init=gamma_init,
                               name='ucov_bn1_2')
        netup1 = conv1.outputs
        # variable_summaries(netup1, 'netup_1')
        # tf.summary.image('netup_1',netup1,10)

        conv1 = tl.layers.Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')
        # tf.summary.image('final', tf.cast(conv1.outputs,tf.uint8), 1)
        # tf.summary.image('final', conv1.outputs, 10)
        conv1 = tf.add(conv1.outputs, inputs.outputs,name = 'output')
        # input = inputs.outputs
        ######## -------------------------Data fidelity--------------------------------##########
        # for contrast in range(n_out):
        #     k_conv3 = utils.Fourier(conv1[:,:,:,contrast], separate_complex=False)
        #     mask = np.ones((batch_size, nx, ny))
        #     mask[:,:, 1:ny:3] = 0
        #     mask = np.fft.ifftshift(mask)
        #     # convert to complex tf tensor
        #     DEFAULT_MAKS_TF = tf.cast(tf.constant(mask), tf.float32)
        #     DEFAULT_MAKS_TF_c = tf.cast(DEFAULT_MAKS_TF, tf.complex64)
        #     k_patches = utils.Fourier(input[:,:,:,contrast], separate_complex=False)
        #     k_space = k_conv3 * DEFAULT_MAKS_TF_c + k_patches*(1-DEFAULT_MAKS_TF_c)
        #     out = tf.ifft2d(k_space)
        #     out = tf.abs(out)
        #     out = tf.reshape(out, [batch_size, nx, ny, 1])
        #     if contrast == 0 :
        #         final_output = out
        #     else:
        #         final_output = tf.concat([final_output,out],3)
        ########-------------------------end------------------------------------###########3
        # print(" * Output: %s" % conv1.outputs)
        # outputs = tl.act.pixel_wise_softmax(conv1.outputs)
        return conv1

def u_net_new(x, n_out=12, reuse = False):  # Do I need to change n_out here ???
    _, nx, ny, nz = x.get_shape().as_list()
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    # gamma_init = tf.random_normal_initializer(1, 0.02)
    with tf.variable_scope("u_net", reuse= reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl.layers.InputLayer(x, name ='inputs')
        conv1 = tl.layers.Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
        pool1 = tl.layers.MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
        # 第二层
        conv2 = tl.layers.Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
        pool2 = tl.layers.MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')

        #第三层
        conv3 = tl.layers.Conv2d(pool2, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_2')
        pool3 = tl.layers.MaxPool2d(conv3, (2, 2), padding='SAME', name='pool3')
        #第四层
        conv4 = tl.layers.Conv2d(pool3, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_2')

        pool4 = tl.layers.MaxPool2d(conv4, (2, 2), padding='SAME', name='pool4')
        # # 第五层
        conv5 = tl.layers.Conv2d(pool4, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_1')
        conv5 = tl.layers.Conv2d(conv5, 1024, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_2')
        up4 = tl.layers.DeConv2d(conv5, 512, (3, 3), out_size=(nx / 8, ny / 8), strides=(2, 2),
                       padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = tl.layers.ConcatLayer([up4, conv4], concat_dim=3, name='concat4')
        conv4 = tl.layers.Conv2d(up4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_1')
        conv4 = tl.layers.Conv2d(conv4, 512, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv4_2')
        #第七层
        up3 = tl.layers.DeConv2d(conv4, 256, (3, 3), out_size=(nx / 4, ny / 4), strides=(2, 2),
                       padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = tl.layers.ConcatLayer([up3, conv3], concat_dim=3, name='concat3')
        conv3 = tl.layers.Conv2d(up3, 256, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_1')
        conv3 = tl.layers.Conv2d(conv3, 256, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='uconv3_2')

        up2 = tl.layers.DeConv2d(conv3, 128, (3, 3), out_size=(nx / 2, ny / 2), strides=(2, 2),
                       padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = tl.layers.ConcatLayer([up2, conv2], concat_dim=3, name='concat2')
        conv2 = tl.layers.Conv2d(up2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_1')
        conv2 = tl.layers.Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv2_2')
        # ###会引起报错的代码
        # netup2 = conv2.outputs
        # variable_summaries(netup2, 'netup_2')

        up1 = tl.layers.DeConv2d(conv2, 64, (3, 3), out_size=(nx / 1, ny / 1), strides=(2, 2),
                       padding='SAME', act=None, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = tl.layers.ConcatLayer([up1, conv1], concat_dim=3, name='concat1')
        conv1 = tl.layers.Conv2d(up1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_1')
        conv1 = tl.layers.Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='uconv1_2')
        # conv1 = BatchNormLayer(conv1, act = lambda x: tl.act.lrelu(x, 0.2), is_train= is_train, gamma_init=gamma_init, name= 'uconv1_bn1')

        conv1 = tl.layers.Conv2d(conv1, n_out, (1, 1), act=None, name='uconv1')

        # conv1 = tf.layers.batch_normalization(inputs =conv1.outputs, axis= -1, training = True)
        # conv1 = tl.layers.InputLayer(conv1, name ='batch_layer9')

        # netup1 = conv1.outputs
        # variable_summaries(netup1, 'netup_1')
        # tf.summary.image('netup_1',netup1,10)

        # print(" * Output: %s" % conv1.outputs)
        #outputs = tl.act.pixel_wise_softmax(conv1.outputs)
        return conv1.outputs
#### DENSE block
# def DesWBH(desBlock_layer, filter_size, outlayer):
#     weightsH = {}
#     biasesH = {}
#     fs = filter_size
#     for i in range(1, outlayer+1):
#         for j in range(1, desBlock_layer+1):
#             if j is 1:
#                 weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, 6, 6], stddev=np.sqrt(2.0/9/16)), name='w_H_%d_%d' % (i, j))})
#             else:
#                 weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, 16* (j-1), 16], stddev=np.sqrt(2.0/9/(16 * (j-1)))), name='w_H_%d_%d' % (i, j))})
#             biasesH.update({'b_H_%d_%d' % (i, j): tf.Variable(tf.zeros([16], name='b_H_%d_%d' % (i, j)))})
#     return weightsH, biasesH
#
#
# def Concatenation(layers):
#     return tf.concat(layers, axis=3)
#
#
# def SkipConnect(conv):
#     skipconv = list()
#     for i in conv:
#         x = Concatenation(i)
#         skipconv.append(x)
#     return skipconv
# def desBlock(input, desBlock_layer, outlayer, filter_size=3 ):
#
#     des_block_H =8
#     des_block_ALL=8
#     weight_block, biases_block = DesWBH(des_block_H, 3, des_block_ALL)
#     w_init = tf.truncated_normal_initializer(stddev=np.sqrt(2.0/9/16))
#     b_init = tf.constant_initializer(value=0.0)
#
#     nextlayer = input
#     conv = list()
#     for i in range(1, outlayer+1):
#         conv_in = list()
#         for j in range(1, desBlock_layer+1):
#             # The first conv need connect with low level layer
#             if j is 1:
#
#                 x = tf.nn.conv2d(nextlayer, weight_block['w_H_%d_%d' %(i, j)], strides=[1,1,1,1], padding='SAME') + biases_block['b_H_%d_%d' % (i, j)]
#                 x = tf.nn.relu(x)
#                 conv_in.append(x)
#             else:
#                 x = Concatenation(conv_in)
#                 x = tf.nn.conv2d(x, weight_block['w_H_%d_%d' % (i, j)], strides=[1,1,1,1], padding='SAME')+ biases_block['b_H_%d_%d' % (i, j)]
#                 x = tf.nn.relu(x)
#                 conv_in.append(x)
#
#         nextlayer = conv_in[-1]
#         conv.append(conv_in)
#     return conv
# def bot_layer(input_layer):
#     # Bottleneck layer
#     des_block_H =8
#     des_block_ALL=8
#     allfeature = 16 * des_block_H * des_block_ALL + 16
#     bot_weight = tf.Variable(tf.random_normal([1, 1, allfeature, 256], stddev=np.sqrt(2.0 / 1 / allfeature)),
#                                   name='w_bot')
#     bot_biases = tf.Variable(tf.zeros([256], name='b_bot'))
#     x = tf.nn.conv2d(input_layer, bot_weight, strides=[1,1,1,1], padding='SAME') + bot_biases
#     x = tf.nn.relu(x)
#     return x
# def reconv_layer( input_layer, n_out):
#     reconv_weight = tf.Variable(tf.random_normal([3, 3, 256, n_out], stddev=np.sqrt(2.0 / 9 / 256)),
#                                      name='w_reconv')
#     reconv_biases = tf.Variable(tf.zeros([6], name='b_reconv'))
#     x = tf.nn.conv2d(input_layer, reconv_weight, strides=[1,1,1,1], padding='SAME') + reconv_biases
#     return x
# def SRDENSE(patches, n_out, is_train, reuse):
#     with tf.variable_scope("SRDENSE", reuse=reuse):
#         tl.layers.set_name_reuse(reuse)
#         x = desBlock(patches, 8, 8, filter_size=3)
#         # NOTE: Cocate all dense block
#
#         x = SkipConnect(x)
#         x.append(patches)
#         x = Concatenation(x)
#         x = bot_layer(x)
#         x = reconv_layer(x, n_out)
#
#     return x
#### DENSE block
def DesWBH(desBlock_layer, filter_size, outlayer):
    weightsH = {}
    biasesH = {}
    fs = filter_size
    for i in range(1, outlayer+1):
        for j in range(1, desBlock_layer+1):
            if j is 1:
                weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, 16, 16], stddev=np.sqrt(2.0/9/16)), name='w_H_%d_%d' % (i, j))})
            else:
                weightsH.update({'w_H_%d_%d' % (i, j): tf.Variable(tf.random_normal([fs, fs, 16* (j-1), 16], stddev=np.sqrt(2.0/9/(16 * (j-1)))), name='w_H_%d_%d' % (i, j))})
            biasesH.update({'b_H_%d_%d' % (i, j): tf.Variable(tf.zeros([16], name='b_H_%d_%d' % (i, j)))})
    return weightsH, biasesH


# def Concatenation(layers,name):
#     return tl.layers.ConcatLayer(layers, concat_dim=3,name=name)
def Concatenation(layers,name):
    return tl.layers.ConcatLayer(layers, concat_dim=3,name=name)

def SkipConnect(conv):
    skipconv = list()

    # x = Concatenation(conv,name='skip')
    # skipconv.append(x)
    k=0
    for i in conv:
        k +=1
        x = Concatenation(i,name = 'skip'+str(k))
        skipconv.append(x)
    return skipconv
def desBlock(input, desBlock_layer, outlayer, filter_size=3, is_train= False ):


    # w_init = tf.truncated_normal_initializer(stddev=np.sqrt(2/9/16))
    w_init = tf.random_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1, 0.02)
    nextlayer = input
    conv = list()
    for i in range(1, outlayer+1):
        conv_in = list()
        for j in range(1, desBlock_layer+1):
            # The first conv need connect with low level layer
            if j is 1:

                x = tl.layers.Conv2d(nextlayer,16,(3,3),act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name ='desblock_'+str(i)+'_'+str(j) )
                # x = BatchNormLayer(x, is_train=is_train, gamma_init=gamma_init,
                #                        name='desblock_'+str(i)+'_'+str(j)+'bn')
                conv_in.append(x)
            else:
                x = Concatenation(conv_in,name = 'concat'+str(i)+'_'+str(j) )
                x = tl.layers.Conv2d(x,16,(3,3),act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name ='desblock_'+str(i)+'_'+str(j))
                # x = BatchNormLayer(x, is_train=is_train, gamma_init=gamma_init,
                #                        name='desblock_'+str(i)+'_'+str(j)+'bn')
                conv_in.append(x)

        nextlayer = conv_in[-1]
        conv.append(conv_in)
    return conv
def bot_layer(input_layer,is_train):
    # Bottleneck layer
    des_block_H =8
    des_block_ALL=8
    allfeature = 16 * des_block_H * des_block_ALL + 16
    # bot_weight = tf.random_normal_initializer(stddev=np.sqrt(2/9/allfeature))
    bot_weight = tf.random_normal_initializer(stddev=0.01)

    bot_biases = tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1, 0.02)
    x = tl.layers.Conv2d(input_layer, 256, (3, 3), act=tf.nn.relu, padding='SAME', W_init=bot_weight, b_init=bot_biases,name = 'bot_layer')
    # x = BatchNormLayer(x, is_train=is_train, gamma_init=gamma_init,
    #                    name='bot_layer_' + 'bn')
    return x
def reconv_layer( input_layer, n_out,is_train):
    # reconv_weight = tf.random_normal_initializer(stddev=np.sqrt(2/9/256))
    reconv_weight = tf.random_normal_initializer(stddev=0.01)

    reconv_biases = tf.constant_initializer(value=0.0)

    gamma_init = tf.random_normal_initializer(1, 0.02)
    x = tl.layers.Conv2d(input_layer, n_out, (3, 3), act=None, padding='SAME', W_init=reconv_weight, b_init=reconv_biases, name='recon_layer')
    # x = BatchNormLayer(x, is_train=is_train, gamma_init=gamma_init,
    #                    name='reconv_layer_' + 'bn')
    return x
def SRDENSE(patches, n_out, is_train, reuse):
    with tf.variable_scope("SRDENSE", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl.layers.InputLayer(patches, name='inputs')
        x = desBlock(inputs, 8, 8, filter_size=3,is_train = is_train)
        # x = Concatenation(x, name='concat')

        x = SkipConnect(x)
        x.append(inputs)

        # NOTE: Cocate all dense block
        x = Concatenation(x,name = 'con_final')
        x = bot_layer(x,is_train)
        x = reconv_layer(x, n_out,is_train)
        final = tf.add(x.outputs, inputs.outputs)
    return final

def srcnn_9751(patches,name ='zqq'):
    with tf.variable_scope(name):
        #upscaled_patches = tf.image.resize_bicubic(patches, [INPUT_SIZE, INPUT_SIZE], True)
        conv1 = conv2d(patches, 9, 9, 64, padding='SAME', name='conv1')
        relu1 = relu(conv1, name='relu1')
        dd = tf.transpose(relu1, perm=[3,1,2,0])
        # tf.summary.image('conv1', dd, 10)

        conv2 = conv2d(relu1, 7, 7, 32, padding='SAME', name='conv2')
        relu2 = relu(conv2, name='relu2')
        dd = tf.transpose(relu2, perm=[3,1,2,0])
        # tf.summary.image('conv2', dd, 10)
        conv3 = conv2d(relu2, 1, 1, 16, padding='SAME', name='conv2')
        relu3 = relu(conv3, name='relu3')

        return conv2d(relu3, 5, 5, NUM_CHENNELS, padding='SAME', name='conv3')

def srcnn_935(patches, name='srcnn',reuse = False):
    batch_size, nx, ny, nz = patches.get_shape().as_list()
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    with tf.variable_scope(name,reuse= reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = tl.layers.InputLayer(patches,name = 'inputs')
        conv1 = tl.layers.Conv2d(inputs,64,(9,9),act = tf.nn.relu, padding='SAME',W_init=w_init,b_init= b_init, name = 'conv1')
        conv2 = tl.layers.Conv2d(conv1,32,(3,3),act = tf.nn.relu, padding='SAME',W_init=w_init,b_init= b_init, name = 'conv2')
        conv3 = tl.layers.Conv2d(conv2,1,(5,5),act = None, padding='SAME',W_init=w_init,b_init= b_init, name = 'conv3')
        #upscaled_patches = tf.image.resize_bicubic(patches, [INPUT_SIZE, INPUT_SIZE], True)
######## Data fidelity
        k_conv3 = utils.Fourier(conv3.outputs,separate_complex = False)
        mask = np.ones((batch_size, nx, ny))
        mask[:,np.array(nx/4+1,dtype= 'int8'):np.array(nx*3/4,dtype= 'int8'),:] = 0
        mask = np.fft.ifftshift(mask)
        # convert to complex tf tensor
        DEFAULT_MAKS_TF = tf.cast(tf.constant(mask), tf.float32)
        DEFAULT_MAKS_TF_c = tf.cast(DEFAULT_MAKS_TF, tf.complex64)
        k_patches = utils.Fourier(patches,separate_complex=False)
        k_space = k_conv3 * DEFAULT_MAKS_TF_c + k_patches
        out = tf.ifft2d(k_space)
        out = tf.abs(out)
        out = tf.reshape(out,[batch_size,nx,ny,1])
        # k_conv3[:,nx/4+1:nx*3/4,ny/4+1:ny*3/4,:] = k_patches[:,nx/4+1:nx*3/4,ny/4+1:ny*3/4,:]

        return out
def SRresnet(t_image, n_out= 12,is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        Input = InputLayer(t_image, name='in')
        n = Conv2d(Input, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(32):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, 'add3')
        # B residual blacks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n256s1/1')
        # n = SubpixelConv2d(n, scale=1, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n256s1/2')
        # n = SubpixelConv2d(n, scale=1, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        # n = Conv2d(n, n_out, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        n = Conv2d(n, n_out, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        n = tf.add(n.outputs, Input.outputs)

        return n
def SRresnet_no_batch(t_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c1/%s' % i)
            # nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                        name='n64s1/c2/%s' % i)
            # nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        # n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, 'add3')
        # B residual blacks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        # n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n256s1/2')
        # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        n = Conv2d(n, NUM_CHENNELS, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, name='out')
        tf.summary.image('final', tf.cast(n.outputs,tf.uint8), 1)
        return n.outputs
def vgg7(patches, name='vgg7'):
    """
    模型的输出
    :param patches: input patches to improve resolution. must has format of
        [batch_size, patch_height, patch_width, patch_chennels]
    :param name: the name of the network
    :return: the RSCNN inference function
    """
    with tf.variable_scope(name):
        #upscaled_patches = tf.image.resize_bicubic(patches, [INPUT_SIZE, INPUT_SIZE], True)
        conv1 = conv2d(patches, 3, 3, 32, padding='SAME', name='conv1')
        lrelu1 = leaky_relu(conv1, name='leaky_relu1')
        conv2 = conv2d(lrelu1, 3, 3, 32, padding='SAME', name='conv2')
        lrelu2 = leaky_relu(conv2, name='leaky_relu2')
        conv3 = conv2d(lrelu2, 3, 3, 64, padding='SAME', name='conv3')
        lrelu3 = leaky_relu(conv3, name='leaky_relu3')
        conv4 = conv2d(lrelu3, 3, 3, 64, padding='SAME', name='conv4')
        lrelu4 = leaky_relu(conv4, name='leaky_relu4')
        conv5 = conv2d(lrelu4, 3, 3, 128, padding='SAME', name='conv5')
        lrelu5 = leaky_relu(conv5, name='leaky_relu5')
        conv6 = conv2d(lrelu5, 3, 3, 128, padding='SAME', name='conv6')
        lrelu6 = leaky_relu(conv6, name='leaky_relu6')
        return conv2d(lrelu6, 3, 3, NUM_CHENNELS, padding='SAME', name='conv_out')


def vgg_deconv_7(patches, name='vgg_deconv_7'):
    with tf.variable_scope(name):
        conv1 = conv2d(patches, 3, 3, 16, padding='SAME', name='conv1')
        lrelu1 = leaky_relu(conv1, name='leaky_relu1')
        conv2 = conv2d(lrelu1, 3, 3, 32, padding='SAME', name='conv2')
        lrelu2 = leaky_relu(conv2, name='leaky_relu2')
        conv3 = conv2d(lrelu2, 3, 3, 64, padding='SAME', name='conv3')
        lrelu3 = leaky_relu(conv3, name='leaky_relu3')
        conv4 = conv2d(lrelu3, 3, 3, 128, padding='SAME', name='conv4')
        lrelu4 = leaky_relu(conv4, name='leaky_relu4')
        conv5 = conv2d(lrelu4, 3, 3, 128, padding='SAME', name='conv5')
        lrelu5 = leaky_relu(conv5, name='leaky_relu5')
        conv6 = conv2d(lrelu5, 3, 3, 256, padding='SAME', name='conv6')
        lrelu6 = leaky_relu(conv6, name='leaky_relu6')

        batch_size = int(lrelu6.get_shape()[0])
        rows = int(lrelu6.get_shape()[1])
        cols = int(lrelu6.get_shape()[2])
        channels = int(patches.get_shape()[3])
        # to avoid chessboard artifacts, the filter size must be dividable by the stride
        return deconv2d(lrelu6, 4, 4, [batch_size, rows, cols, channels], stride=(1, 1), name='deconv_out')
def EDSR(patches,feature_size=64,num_layers=16):
    print("Building EDSR...")
    # mean_x = tf.reduce_mean(patches)
    # image_input = patches - mean_x
    # mean_y = tf.reduce_mean(high_patches)
    # image_target = y - mean_y

    x = slim.conv2d(patches, feature_size, [3, 3])
    conv_1 = x


    """
            This creates `num_layers` number of resBlocks
            a resBlock is defined in the paper as
            (excuse the ugly ASCII graph)
            x
            |\
            | \
            |  conv2d
            |  relu
            |  conv2d
            | /
            |/
            + (addition here)
            |
            result
            """

    """
    Doing scaling here as mentioned in the paper:
    
    `we found that increasing the number of feature
    maps above a certain level would make the training procedure
    numerically unstable. A similar phenomenon was
    reported by Szegedy et al. We resolve this issue by
    adopting the residual scaling with factor 0.1. In each
    residual block, constant scaling layers are placed after the
    last convolution layers. These modules stabilize the training
    procedure greatly when using a large number of filters.
    In the test phase, this layer can be integrated into the previous
    convolution layer for the computational efficiency.'
    
    """
    scaling_factor = 1

    # Add the residual blocks to the model
    for i in range(num_layers):
        x = resBlock(x, feature_size, scale=scaling_factor)

    # One more convolution, and then we add the output of our first conv layer
    x = slim.conv2d(x, feature_size, [3, 3])
    x += conv_1

    # Upsample output of the convolution
    x = upsample(x, NUM_CHENNELS, Scale,feature_size, None)
    output =x
    mean_x = tf.reduce_mean(patches)
    f = tf.sqrt(tf.reduce_sum(tf.square(output + mean_x), axis=-1))
    ff = tf.reshape(f, [-1, 384, 384, 1])
    tf.summary.image("output_image", tf.cast(ff, tf.uint8))
    # f = tf.sqrt(tf.reduce_sum(tf.square(output[:,:,:,0:2] + mean_x), axis=-1))
    # ff = tf.reshape(f, [-1, 384, 384, 1])
    # tf.summary.image("output_image_echo1_FA1", tf.cast(ff, tf.uint8))
    # f = tf.sqrt(tf.reduce_sum(tf.square(output[:,:,:,2:4] + mean_x), axis=-1))
    # ff = tf.reshape(f, [-1, 384, 384, 1])
    # tf.summary.image("output_image_echo2_FA1", tf.cast(ff, tf.uint8))
    # f = tf.sqrt(tf.reduce_sum(tf.square(output[:,:,:,4:6] + mean_x), axis=-1))
    # ff = tf.reshape(f, [-1, 384, 384, 1])
    # tf.summary.image("output_image_echo3_FA1", tf.cast(ff, tf.uint8))
    # f = tf.sqrt(tf.reduce_sum(tf.square(output[:,:,:,6:8] + mean_x), axis=-1))
    # ff = tf.reshape(f, [-1, 384, 384, 1])
    # tf.summary.image("output_image_echo1_FA2", tf.cast(ff, tf.uint8))
    # f = tf.sqrt(tf.reduce_sum(tf.square(output[:,:,:,8:10] + mean_x), axis=-1))
    # ff = tf.reshape(f, [-1, 384, 384, 1])
    # tf.summary.image("output_image_echo2_FA2", tf.cast(ff, tf.uint8))
    # f = tf.sqrt(tf.reduce_sum(tf.square(output[:,:,:,10:12] + mean_x), axis=-1))
    # ff = tf.reshape(f, [-1, 384, 384, 1])
    # tf.summary.image("output_image_echo3_FA2", tf.cast(ff, tf.uint8))
    return output

def loss(inferences, ground_truthes, huber_width=0.1, weights_decay=0, name='loss'):
    with tf.name_scope(name):


        # delta *= [[[[0.11448, 0.58661, 0.29891]]]]  # weights of B, G and R
        # delta *= [1]
        # l2_loss = tf.pow(delta, 2)
        # mse_loss = tf.reduce_mean(tf.reduce_sum(l2_loss, axis=[1, 2, 3]))
        # mse_loss = tf.reduce_mean(tf.reduce_sum(l2_loss))
        mse_loss = tf.reduce_mean(tf.square(inferences - ground_truthes))


        return mse_loss
def loss_l1(inferences, ground_truthes, huber_width=0.1, weights_decay=0, name='loss'):
    with tf.name_scope(name):
        # mean_y = tf.reduce_mean(ground_truthes)
        # image_target = ground_truthes - mean_y
        # delta *= [[[[0.11448, 0.58661, 0.29891]]]]  # weights of B, G and R
        # delta *= [1]
        # l2_loss = tf.pow(delta, 2)
        # mse_loss = tf.reduce_mean(tf.reduce_sum(l2_loss, axis=[1, 2, 3]))
        # mse_loss = tf.reduce_mean(tf.reduce_sum(l2_loss))
        mse_loss = tf.reduce_mean(tf.losses.absolute_difference(ground_truthes, inferences))


        return mse_loss

def PSNRLoss(inferences,ground_truthes, name = 'accurancy'):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    with tf.name_scope(name):
        accurancy = tf.reduce_mean(-10. * tf.log(tf.square(inferences-ground_truthes)))

    return accurancy
# def ssim_loss(inferences, ground_truthes, BATCH_SIZE, PATCH_SIZE, Number_CHANNELS = NUM_CHENNELS, name  = 'ssim_loss'):
#     img1, img2 = Norm_to_gray_scale(inferences,ground_truthes,BATCH_SIZE,PATCH_SIZE,Number_CHANNELS )
#     with tf.name_scope(name):
#         ssim_value = tf_ssim(img1,img2)
#         ssim_value2 = loss_DSSIS_tf11(img1,img2)
#         return  ssim_value, ssim_value2
