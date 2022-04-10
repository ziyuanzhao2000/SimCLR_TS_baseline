# Convert physionet2017 train and test datasets from .mat + .csv files to
# dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}
# to be taken by data_pre_processing.pre_process_dataset_composite

import numpy as np
import os
import pickle
import wfdb
import csv
import data_pre_processing
from tqdm import tqdm

data_folder = 'test_run'
aliases = ['training2017', 'validation']

sampling_fs = 300 # Hz
window_len = 1500 # samples
validation_prefix = 'V'  # for disambiguating patient id in validation set from those in the training set
dataset = {}

for alias in aliases:
    basepath = f'{os.getcwd()}/{data_folder}/{alias}'
    # First get list of file names from the data folder
    file_names = [file_name.split('.hea')[0] for file_name in os.listdir(basepath) if '.hea' in file_name]
    diagnoses = []
    with open(os.path.join(basepath, 'REFERENCE.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            diagnoses.append(line[1])

    # Process each file by dividing up time series into contiguous windows
    for file_name, diagnosis in tqdm(zip(file_names, diagnoses)):
        signal, _ = wfdb.rdsamp(os.path.join(basepath, file_name))
        if alias == 'validation':
            file_name = validation_prefix + file_name
        assert(signal.shape[0] > window_len)
        dataset[file_name] = [(signal, np.array([diagnosis]*len(signal)))] # file_name is just

label_map = {'N':0, 'A':1, 'O':2, '~':3}

train, val, test = data_pre_processing.pre_process_dataset_composite(
    user_datasets=dataset,
    label_map=label_map,
    train_users=[username for username in dataset.keys() if not validation_prefix in username],
    test_users=[username for username in dataset.keys() if validation_prefix in username],
    output_shape=len(label_map),
    window_size=window_len,
    shift=window_len # no overlap
)

train_x, train_y = train
val_x, val_y = val
test_x, test_y = test

np.save(os.path.join(data_folder, 'train_x'), train_x)
np.save(os.path.join(data_folder, 'train_y'), train_y)
np.save(os.path.join(data_folder, 'val_x'), val_x)
np.save(os.path.join(data_folder, 'val_y'), val_y)
np.save(os.path.join(data_folder, 'test_x'), test_x)
np.save(os.path.join(data_folder, 'test_y'), test_y)

