import tensorflow as tf
import numpy as np


# input roi_features_4d: [100, 7, 7, 512]


class FcTail(object):

    def __init__(self, roi_features_4d):
        with tf.variable_scope('FcTail'):
            self.roi_features_1d = tf.reshape(tensor=roi_features_4d, shape=[-1, 25088])

            self.bbox = tf.contrib.layers.fully_connected(
                inputs=self.roi_features_1d,
                num_outputs=4
            )

            self.softmax_input = tf.contrib.layers.fully_connected(
                inputs=self.roi_features_1d,
                num_outputs=20
            )

            self.softmax_output = tf.nn.softmax(self.softmax_input)

    def inference(self):
        return self.bbox, self.softmax_output


if __name__ == '__main__':
    fake_input = tf.random_normal(shape=[100, 7, 7, 512])
    fc_instant = FcTail(fake_input)
    a = fc_instant.inference()
    print(a)
