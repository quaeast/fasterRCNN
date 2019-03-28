import tensorflow as tf
import numpy as np

# input
# feature map: [1, 38, 63, 512]
# roi boxes: [n, 4]
# output [n, 7, 7, 512]

def roi_pool(feature_map, rois):
    with tf.variable_scope('roi_pool'):
        box_ind = tf.zeros(shape=rois.shape[0], dtype=tf.int32)
        crop_size = tf.constant(value=[14, 14])
        roi_features = tf.image.crop_and_resize(image=feature_map, boxes=rois, box_ind=box_ind, crop_size=crop_size)
        pooled_roi_features = tf.nn.max_pool(value=roi_features, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pooled_roi_features





if __name__ == '__main__':
    features = tf.random_normal(shape=[1, 38, 63, 512], dtype=tf.float32)
    rois = tf.random_uniform(shape=[100, 4], minval=0, maxval=1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a = roi_pool(features, rois).shape
        print(a)

