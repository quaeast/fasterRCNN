import tensorflow as tf
import numpy as np


nparray = np.array([1, 2])



# with tf.variable_scope(name_or_scope="direct"):
#     d_a = tf.get_variable(name='a', initializer=nparray)


with tf.variable_scope(name_or_scope="use_init"):
    c = tf.constant_initializer(value=nparray, dtype=tf.float32)
    u_a = tf.get_variable(name='a', initializer=c, shape=nparray.shape)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(d_a))
    print(sess.run(u_a))

