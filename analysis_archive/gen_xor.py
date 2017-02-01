import h5py as h5
import numpy as np

data_X = np.random.normal([0,0], [1,1], size=[10000,2])
data_Y = np.logical_xor(
    data_X[:,0] > 0,
    data_X[:,1] > 0
).astype(np.float32)

test_X = np.random.normal([0,0], [1,1], size=[1000,2])
test_Y = np.logical_xor(
    test_X[:,0] > 0,
    test_X[:,1] > 0
).astype(np.float32)

hfile = h5.File('xor.h5')
hfile.create_dataset('training/inputs', data=data_X)
hfile.create_dataset('training/labels', data=data_Y)
hfile.create_dataset('validation/inputs', data=test_X)
hfile.create_dataset('validation/labels', data=test_Y)
hfile.close()
