import numpy as np

tr = np.load('full/train.npy')
va = np.load('full/val.npy')
te = np.load('full/test.npy')

window_len = 178
tr = tr[:,:window_len,0:1]
va = va[:,:window_len,0:1]
te = te[:,:window_len,0:1]

np.save('train.npy', tr)
np.save('val.npy', va)
np.save('test.npy', te)