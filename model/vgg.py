import tensorflow as tf
import numpy as np


class Vgg16(object):

    def __init__(self, vgg_npy_path):
        self.model = np.load(file=vgg_npy_path, encoding='latin1').item()
        print('vgg16.npy loaded')

    def get_filter(self, name):
        con_filter = self.model[name][0]
        return tf.get_variable(name='filter', initializer=con_filter)

    def get_weight(self, name):
        weight = self.model[name][0]
        return tf.get_variable(name='weight', initializer=weight)

    def get_bias(self, name):
        bias = self.model[name][1]
        return tf.get_variable(name='bias', initializer=bias)

    def conv2D(self, input_x, name):
        with tf.variable_scope(name_or_scope=name):
            con_filter = self.get_filter(name)
            bias = self.get_bias(name)
            input_x = tf.nn.conv2d(input=input_x, filter=con_filter, strides=[1, 1, 1, 1], padding='SAME')
            input_x = tf.nn.bias_add(value=input_x, bias=bias)
            return tf.nn.relu(features=input_x)

    def max_pool(self, input_x, name):
        with tf.variable_scope(name_or_scope=name):
            return tf.nn.max_pool(value=input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def convs(self, img):

        conv1_1 = self.conv2D(img, 'conv1_1')
        conv1_2 = self.conv2D(self.conv1_1, 'conv1_2')
        max_pool1 = self.max_pool(self.conv1_2, 'max_pool1')

        conv2_1 = self.conv2D(self.max_pool1, 'conv2_1')
        conv2_2 = self.conv2D()




