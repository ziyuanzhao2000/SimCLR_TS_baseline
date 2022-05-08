import numpy as np

tr = np.load('full/train_x.npy')
va = np.load('full/val_x.npy')
te = np.load('full/test_x.npy')

window_len = 178
tr = tr[:,:window_len,0:1]
va = va[:,:window_len,0:1]
te = te[:,:window_len,0:1]

np.save('train_x.npy', tr)
np.save('val_x.npy', va)
np.save('test_x.npy', te)