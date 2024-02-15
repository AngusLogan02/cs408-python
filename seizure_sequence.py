import tensorflow as tf
from data import get_seizure_data
import numpy as np
import math
from preprocessor import PreProcessor
import os

class SeizureSequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size, dataset, cases, bias_positive, shuffle=True, preprocessor: PreProcessor = None):
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.cases = cases
        self.bias_positive = bias_positive
        self.preprocessor = preprocessor

        files = [filename for filename in os.listdir(dataset) if os.path.basename(filename).split('_')[0] in self.cases]
        files = ['_'.join(filename.split('_')[:-1]) for filename in files]

        self.files = files
        self.test_file = files[0]
        self.files = self.files[1:]
        self.file_count = len(self.files)

    def __len__(self):
        return math.ceil(self.file_count / self.batch_size)
    
    def get_files(self):
        return self.files
    
    def get_test_file(self):
        return self.test_file

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, self.file_count)

        first_run = True
        for i in range(low, high):
            data = np.load(self.dataset + "/" + self.files[i] + "_data.npy")
            labels = np.load(self.dataset + "/" + self.files[i] + "_labels.npy")

            if self.preprocessor is not None:
                approx, detail = self.preprocessor.transform(data)
                data = detail
                data = self.preprocessor.extract_features(data)
            if first_run:
                batch_data = data
                batch_labels = labels
                first_run = False
            else:
                batch_data = np.append(batch_data, data, axis=0)
                batch_labels = np.append(batch_labels, labels, axis=0)

        return batch_data, batch_labels
