import model.vgg
import tensorflow as tf
import numpy as np


class RPN(object):

    def __init__(self, feature_map):
        anchor_num = 9
        with tf.variable_scope('rpn_head'):

            self.conv3m3 = tf.layers.conv2d(inputs=feature_map,
                                            filters=512,
                                            kernel_size=[3, 3],
                                            padding='same',
                                            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                            name='rpn_conv_3m3')

            self.rpn_cls = tf.layers.conv2d(inputs=self.conv3m3,
                                            filters=anchor_num,
                                            kernel_size=[1, 1],
                                            padding='same',
                                            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                            name='rpn_cls')

            self.rpn_reg = tf.layers.conv2d(inputs=self.conv3m3,
                                            filters=anchor_num * 4,
                                            kernel_size=[1, 1],
                                            padding='same',
                                            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                                            name='rpn_reg')

    def pred(self):
        return self.rpn_cls, self.rpn_reg


if __name__ == '__main__':
    vgg = model.vgg.Vgg16('../vgg_data/vgg16.npy')
    net = RPN(vgg.get_conv_result())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(fetches=net.pred(), feed_dict={vgg.img: np.ones([1, 224, 224, 3])})
