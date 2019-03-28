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

        with tf.variable_scope('anchor_normalize'):
            rpn_reg_anchor = tf.math.divide(rpn_reg_anchor, tf.constant(value=divisor, dtype=tf.float32))
            rpn_reg_anchor = tf.clip_by_value(rpn_reg_anchor, clip_value_min=0, clip_value_max=1)
    return rpn_reg_anchor


# mode:
#   train ->'t' output shape 12000 2000
#   inference -> 'r' output shape 6000 300
#

def nmp(rpn_reg_anchor, rpn_reg_scores, mode='t'):
    if mode == 't':
        top_n = 12000
        max_output = 2000
    else:
        top_n = 6000
        max_output = 300
    with tf.name_scope('nmp'):
        rpn_reg_anchor_2d = tf.reshape(rpn_reg_anchor, shape=[-1, 4])
        rpn_reg_scores_1d = tf.reshape(rpn_reg_scores, shape=[-1])
        suppressed_anchors_index = tf.image.non_max_suppression(boxes=rpn_reg_anchor_2d,
                                                          scores=rpn_reg_scores_1d,
                                                          max_output_size=max_output)
        suppressed_anchors = tf.gather(params=rpn_reg_anchor_2d, indices=suppressed_anchors_index)
        return suppressed_anchors


if __name__ == '__main__':
    fake_anchors = tf.random_normal(dtype=tf.float32, shape=[1, 38, 63, 36])
    fake_scores = tf.random_normal(dtype=tf.float32, shape=[1, 38, 63, 9])
    sa = nmp(generate_anchor(fake_anchors), fake_scores)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        na = sess.run(sa)
        print(na.shape)
