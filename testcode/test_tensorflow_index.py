import tensorflow as tf
import numpy as np

a = np.array(range(10))
b = np.array(range(0, 10, 2))

a += 20

tfa = tf.get_variable(initializer=a, name='tfa')
tfb = tf.get_variable(initializer=b, name='tfb')
tfc = [tfb[i] for i in tfb]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out = sess.run(tfc)
    print(out)
