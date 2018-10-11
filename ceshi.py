import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph
# 本来这个model本无需解释太多，但是这么多人不能耐下心来看，那么我简单的说一下吧
# network是你们自己定义的模型结构而已
# ps:
# def network(input)：
#    return tf.layers.max_pooling2d(input, 2, 2)
import SRCNN_models


os.environ['CUDA_VISIBLE_DEVICES']='2'  #设置GPU


model_path  =  "model_Amp_6channel_bn_7_12//u_net_bn_new_2//good//mymodel" #设置model的路径，因新版tensorflow会生成三个文件，只需写到数字前


def main():

    tf.reset_default_graph()

    input_node = tf.placeholder(tf.float32, shape=(228, 304, 3)) #这个是你送入网络的图片大小，如果你是其他的大小自行修改
    input_node = tf.expand_dims(input_node, 0)
    flow = SRCNN_models.create_model('u_net_bn_new2', input_node, n_out=6, is_train=False)
    # flow = tf.cast(flow, tf.uint8, 'out') #设置输出类型以及输出的接口名字，为了之后的调用pb的时候使用

    saver = tf.train.Saver()
    with tf.Session() as sess:

        saver.restore(sess, model_path)

        #保存图
        tf.train.write_graph(sess.graph_def, 'output_model/pb_model', 'model.pb')
        #把图和参数结构一起
        freeze_graph.freeze_graph('output_model/pb_model/model.pb', '', False, model_path, 'out','save/restore_all', 'save/Const:0', 'output_model/pb_model/frozen_model.pb', False, "")

    print("done")

if __name__ == '__main__':
    main()