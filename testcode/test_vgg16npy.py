import numpy as np

a = np.load(file='../vgg_data/vgg16.npy', encoding='latin1').item()

conv_filter = a['conv2_1'][0]

print(conv_filter.shape)

