import numpy as np

a = np.load(file='vgg16.npy', encoding='latin1').item()

conv_filter = a['conv2_1'][1]

print(conv_filter.shape)

