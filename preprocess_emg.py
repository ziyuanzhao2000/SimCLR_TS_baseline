
import os
import numpy as np
import data_pre_processing

sample_fs = 4000 # Hz
target_fs = 4000 # Hz
downsample_stride = sample_fs // target_fs
target_fs = sample_fs // downsample_stride
window_len = 1500 # samples

# first stage, read in datasets and make the dictionaries for X (id -> time series) and y (id -> patient no.)
data_folder = 'test_run'
alias = 'emg'
data_file_names = ['emg_healthy.txt', 'emg_myopathy.txt', 'emg_neuropathy.txt']
labels = [0, 1, 2]

basepath = os.path.join(os.getcwd(), data_folder, alias)
dataset = {}
pid = 0
for data_file_name, label in zip(data_file_names, labels):
    signal = np.loadtxt(os.path.join(basepath, data_file_name))[:,1:2] # first column is timestamp, not needed
    signal = signal[::downsample_stride,:] # downsample
    signal_length = signal.shape[0]
    signal = signal[:signal_length // window_len * window_len,:]
    signal_length = signal.shape[0]
    signals = signal.reshape((signal_length // window_len, window_len))
    print(signals.shape)
    for i in range(signals.shape[0]):
        dataset[pid] = [(signals[i:i+1, :].T, np.array([label] * window_len))]
        pid += 1

label_map = {0:0, 1:1, 2:2} # trivial
train_proportion = 0.8
test_proportion = 1 - train_proportion
n_samples = pid

train_users = np.random.choice(np.arange(n_samples), int(train_proportion * n_samples), replace=False)
test_users = np.setdiff1d(np.arange(n_samples), train_users)

train, val, test = data_pre_processing.pre_process_dataset_composite(
    user_datasets=dataset,
    label_map=label_map,
    train_users=train_users,
    test_users=test_users,
    output_shape=len(label_map),
    window_size=window_len,
    shift=window_len # no overlap
)

train_x, train_y = train
val_x, val_y = val
test_x, test_y = test

np.save(os.path.join(data_folder, alias, 'train_x'), train_x)
np.save(os.path.join(data_folder, alias, 'train_y'), train_y)
np.save(os.path.join(data_folder, alias, 'val_x'), val_x)
np.save(os.path.join(data_folder, alias, 'val_y'), val_y)
np.save(os.path.join(data_folder, alias, 'test_x'), test_x)
np.save(os.path.join(data_folder, alias, 'test_y'), test_y)
