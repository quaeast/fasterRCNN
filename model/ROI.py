import tensorflow as tf
import numpy as np


class ROI(object):

    def __init__(self, pooling_size, feature_map, rois):
        self.pooling_size = pooling_size
        