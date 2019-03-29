import tensorflow as tf
import numpy as np
import model.vgg as vgg
import model.rpn as rpn
import model.FcTail as FcTail
import lib.generate_anchor as lig
import lib.ROI as libr


vgg = vgg.Vgg16('../vgg_data/vgg16.npy')
feature_map = vgg.get_feature_map()
rpn_cls, rpn_reg = rpn.RPN(feature_map).pred()
roi_boxes = lig.generate_anchor(rpn_reg)
roi_boxes_reduced = lig.nms(roi_boxes, rpn_cls)
# print(feature_map.shape)
# print(roi_boxes_reduced.shape)
roi_feature_maps = libr.roi_pool(feature_map, roi_boxes_reduced)
fc = FcTail.FcTail(roi_feature_maps)
box_reg, cls = fc.inference()

with tf.Session() as sess:
    fake_image = np.random.uniform(size=[1, 600, 1000, 3], low=0, high=100)
    sess.run(tf.global_variables_initializer())
    sess.run(box_reg, feed_dict={vgg.img: fake_image})
