import tensorflow as tf
import numpy as np


def smooth_l1(x):
    condition = tf.less(tf.abs(x), tf.ones(shape=x.shape))
    con_t = 0.5*x**2
    con_f = tf.abs(x)-0.5
    return tf.where(condition, con_t, con_f)


def loss(cls, cls_t, bbox, bbox_t):
    cls_loss = tf.reduce_mean(smooth_l1(cls-cls_t))
    bbox_loss = tf.reduce_mean(smooth_l1(bbox-bbox_t))
    return cls_loss + bbox_loss

