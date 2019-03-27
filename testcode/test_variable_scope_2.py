import tensorflow as tf
import numpy as np

with tf.variable_scope('foo'):
    a = tf.get_variable(name='a', initializer=np.ones([2, 3, 4]))
    a = tf.reshape(tensor=a, shape=[2, 12], name='a')


print(a.name)
# print(b.name)