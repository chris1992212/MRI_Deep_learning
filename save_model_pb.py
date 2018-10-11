import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import SRCNN_models
import os
from tensorflow.python.framework import graph_util

# os.environ['CUDA_VISIBLE_DEVICES']='2'  #设置GPU
# model_path = "D:/SR_crop/equal abs_mae_ssim/abs.ckpt.meta"  # 设置model的路径
restore_model_filename = os.path.join('model_Amp_6channel_bn_7_12','u_net_bn_new_2','good','mymodel')
def main():
    tf.reset_default_graph()
    low_res_holder = tf.placeholder(tf.float32, shape=[None, None, None, 6], name = 'low')
    inferences = SRCNN_models.create_model('u_net_bn_new_2', low_res_holder,n_out= 6,is_train= False)
    saver = tf.train.import_meta_graph("D://SR_crop//model_Amp_6channel_bn_7_12//u_net_bn_new_2//good//mymodel.meta")
    with tf.Session() as sess:
        saver.restore(sess, restore_model_filename)

        # 保存图
        tf.train.write_graph(sess.graph_def, './pb_dir_output/pb_model', 'mymodel_final.pbtxt')
        # 把图和参数结构一起
        freeze_graph.freeze_graph('pb_dir_output/pb_model/mymodel_final.pbtxt',
                                  '',
                                  False,
                                  restore_model_filename,
                                  'u_net/output',
                                  'save/restore_all',
                                  'save/Const:0',
                                  'pb_dir_output/pb_model/model_frozen.pb',
                                  False,
                                  "")
    print("done")


if __name__ == '__main__':
    main()
    tf.nn.conv2d_transpose