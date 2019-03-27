import tensorflow as tf
import numpy as np


class Vgg16(object):

    def __init__(self, vgg_npy_path):
        self.model = np.load(file=vgg_npy_path, encoding='latin1').item()
        print('vgg16.npy loaded')

        self.img = tf.placeholder(shape=[None, 600, 1000, 3], dtype=tf.float32)

        self.conv1_1 = self.conv2D(self.img, 'conv1_1')
        self.conv1_2 = self.conv2D(self.conv1_1, 'conv1_2')
        self.max_pool1 = self.max_pool(self.conv1_2, 'max_pool1')

        self.conv2_1 = self.conv2D(self.max_pool1, 'conv2_1')
        self.conv2_2 = self.conv2D(self.conv2_1, 'conv2_2')
        self.max_pool2 = self.max_pool(self.conv2_2, 'max_pool2')

        self.conv3_1 = self.conv2D(self.max_pool2, 'conv3_1')
        self.conv3_2 = self.conv2D(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv2D(self.conv3_2, 'conv3_3')
        self.max_pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv2D(self.max_pool3, 'conv4_1')
        self.conv4_2 = self.conv2D(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv2D(self.conv4_2, 'conv4_3')
        self.max_pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv2D(self.max_pool4, 'conv5_1')
        self.conv5_2 = self.conv2D(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv2D(self.conv5_2, 'conv5_3')

        # self.conv5_4.shape=(?, 38, 63, 512)

        # self.max_pool5 = self.max_pool(self.conv5_3, 'max_pool5')
        #
        # self.max_pool5 = tf.reshape(self.max_pool5, [-1, 25088])
        #
        # self.fc6 = self.full_connect(self.max_pool5, 'fc6')
        #
        # self.fc7 = self.full_connect(self.fc6, 'fc7')
        #
        # self.fc8 = self.full_connect(self.fc7, 'fc8')
        #
        # self.softmax = tf.nn.softmax(self.fc8)


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

    def full_connect(self, input_x, name):
        with tf.variable_scope(name_or_scope=name):
            weight = self.get_weight(name)
            bias = self.get_bias(name)
            return tf.nn.relu_layer(x=input_x, weights=weight, biases=bias)

    def get_feature_map(self):
        return self.conv5_3

    # def get_softmax(self):
    #     return self.softmax


if __name__ == '__main__':
    with tf.Session() as sess:
        vgg = Vgg16('../vgg_data/vgg16.npy')
        sess.run(tf.global_variables_initializer())
        print('initialized')
        print(sess.run(fetches=vgg.get_feature_map(), feed_dict={vgg.img: np.ones(dtype=np.float32, shape=[1, 600, 1000, 3])}))


'''
CPU times: user 19.2 s, sys: 15.9 s, total: 35.1 s
Wall time: 38.3 s
'''