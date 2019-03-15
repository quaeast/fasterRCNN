import tensorflow as tf
import numpy as np


a = tf.get_variable(name='w', initializer=np.array([1]))

def get_w():
    return tf.get_variable(name='w', initializer=np.array([1]))


with tf.variable_scope(name_or_scope='fang'):
    b = get_w()
    c = a

print(b.name)
print(c.name)
