import tensorflow as tf
import numpy as np

anchor_map = np.array([[-84, -40, 99, 55],
                       [-176, -88, 191, 103],
                       [-360, -184, 375, 199],
                       [-56, -56, 71, 71],
                       [-120, -120, 135, 135],
                       [-248, -248, 263, 263],
                       [-36, -80,   51, 95],
                       [-80, -168, 95, 183],
                       [-168, -344, 183, 359]], dtype='float32')

anchor_map_1d = np.reshape(anchor_map, [-1])

divisor = np.zeros(shape=[36], dtype='float32')

divisor[::2] = 600
divisor[1::2] = 1000

# 600*1000
# rpn_reg.shape()
# (?, 38, 63, 9*4)

# y1, x1, y2, x2



def generate_anchor(rpn_reg):
    with tf.variable_scope('generate_anchor'):
        anchor_map_1d_tfc = tf.constant(value=anchor_map_1d, dtype=tf.float32)
        rpn_reg_anchor = tf.math.add(rpn_reg, anchor_map_1d_tfc)
        print(rpn_reg_anchor)
        with tf.variable_scope('anchor_normalize'):
            rpn_reg_anchor_t = tf.math.divide(rpn_reg_anchor[::2], tf.constant(value=divisor, dtype=tf.float32))
    return rpn_reg_anchor_t


if __name__ == '__main__':
    fake_anchors = tf.random_normal(dtype=tf.float32, shape=[1, 38, 63, 36])
    fake_scores = tf.random_normal(dtype=tf.float32, shape=[1, 38, 63, 9])
    sa = generate_anchor(fake_anchors)
    print(sa.shape)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     na = sess.run(sa)
    #     print(na.shape)
