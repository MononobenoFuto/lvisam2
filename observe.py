import numpy as np

filename = '/media/nyamori/8856D74A56D73820/vslam/dataset/kitti/2011_09_30/flow/0000000000.npy'
data = np.load(filename)
print(data.shape)