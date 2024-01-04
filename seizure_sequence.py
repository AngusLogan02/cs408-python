import tensorflow as tf
from data import get_seizure_data
import numpy as np
import math

class SeizureSequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size, dataset, cases, bias_positive, shuffle=True):
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.cases = cases
        self.bias_positive = bias_positive

        files = []
        for case in self.cases:
            case_data = get_seizure_data(case)
            for seizure_data in case_data:
                files.append(seizure_data.filename.split("/")[-1])

        self.files = files
        self.file_count = len(files)

    def __len__(self):
        return math.ceil(self.file_count / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, self.file_count)

        if self.bias_positive:
            data = np.empty((0, 23, 256))
            labels = np.empty((0, 1))
            for i in range(low, high):
                orig_data = np.load(self.dataset + "/" + self.files[i] + "_data.npy")
                orig_labels = np.load(self.dataset + "/" + self.files[i] + "_labels.npy")

                positive_count = np.count_nonzero(orig_labels)
                if positive_count == 0:
                     continue
                positive_indices = []
                for j in range(0, len(orig_labels)):
                    if orig_labels[j][0] == 1:
                        positive_indices.append(j)

                negative_indices = []
                for j in range(0, len(orig_labels)):
                    if orig_labels[j][0] == 0:
                        negative_indices.append(j)
                    if len(negative_indices) == positive_count:
                        break
                    
                positive_data = orig_data[positive_indices[0]:positive_indices[-1]]
                positive_labels = orig_labels[positive_indices[0]:positive_indices[-1]]
                negative_data = orig_data[negative_indices[0]:negative_indices[-1]]
                negative_labels = orig_labels[negative_indices[0]:negative_indices[-1]]
                
                file_data = np.concatenate((positive_data, negative_data), axis=0)
                file_labels = np.concatenate((positive_labels, negative_labels), axis=0)
                data = np.concatenate((data, file_data), axis=0)
                labels = np.concatenate((labels, file_labels), axis=0)

            return data, labels
        
        else:
            for i in range(low, high):
                data = np.load(self.dataset + "/" + self.files[i] + "_data.npy")
                labels = np.load(self.dataset + "/" + self.files[i] + "_labels.npy")
            
                return np.array(data), np.array(labels)
            


class SeizureSequence2(tf.keras.utils.Sequence):
    def __init__(self, batch_size, dataset, cases, bias_positive, shuffle=True):
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.cases = cases
        self.bias_positive = bias_positive

        files = []
        for case in self.cases:
            case_data = get_seizure_data(case)
            for seizure_data in case_data:
                files.append(seizure_data.filename.split("/")[-1])

        self.files = files
        self.file_count = len(files)

    def __len__(self):
        return math.ceil(self.file_count / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, self.file_count)

        if self.bias_positive:
            data = []
            labels = []
            files = []
            for i in range(low, high):
                files.append(self.files[i])
                orig_data = np.load(self.dataset + "/" + self.files[i] + "_data.npy")
                orig_labels = np.load(self.dataset + "/" + self.files[i] + "_labels.npy")

                positive_count = np.count_nonzero(orig_labels)
                if positive_count == 0:
                    continue

                positive_indices = [j for j, label in enumerate(orig_labels) if label[0] == 1]
                negative_indices = [j for j, label in enumerate(orig_labels) if label[0] == 0][:positive_count]

                positive_data = orig_data[positive_indices[0]:positive_indices[-1]]
                positive_labels = orig_labels[positive_indices[0]:positive_indices[-1]]
                negative_data = orig_data[negative_indices[0]:negative_indices[-1]]
                negative_labels = orig_labels[negative_indices[0]:negative_indices[-1]]

                file_data = np.concatenate((positive_data, negative_data), axis=0)
                file_labels = np.concatenate((positive_labels, negative_labels), axis=0)
                data.append(file_data)
                labels.append(file_labels)

            data = np.concatenate(data, axis=0)
            labels = np.concatenate(labels, axis=0)
            print("Sending", len(files), "files", files)
            return data, labels

        else:
            data = []
            labels = []
            for i in range(low, high):
                orig_data = np.load(self.dataset + "/" + self.files[i] + "_data.npy")
                orig_labels = np.load(self.dataset + "/" + self.files[i] + "_labels.npy")
                data.append(orig_data)
                labels.append(orig_labels)

            data = np.concatenate(data, axis=0)
            labels = np.concatenate(labels, axis=0)

            return data, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.files)

    def create_tf_dataset(self):
        return tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                tf.TensorSpec(shape=(None, 23, 256), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            ),
        ).prefetch(tf.data.experimental.AUTOTUNE)

    def generator(self):
        for i in range(len(self)):
            yield self[i]

# # Usage
# batch_size = 2
# dataset = "your_dataset_path"
# cases = "your_cases"
# bias_positive = True
# shuffle = True

# seizure_sequence = SeizureSequence(batch_size, dataset, cases, bias_positive, shuffle)
# tf_dataset = seizure_sequence.create_tf_dataset()

# model.fit(tf_dataset, epochs=10)
