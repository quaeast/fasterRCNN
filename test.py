import numpy as np

a = np.load(file='vgg16.npy', encoding='latin1').item()

for i in a:
    print(i)

