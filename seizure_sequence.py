from typing import List, Tuple
import tensorflow as tf
from seizure_data import get_seizure_data
import numpy as np
from numpy.typing import NDArray
import math
import os

class SeizureSequence(tf.keras.utils.Sequence):
    """A custom Keras Sequence, used to supply EEG data and labels to the model."""

    def __init__(self, batch_size, dataset, cases) -> None:
        """Initialises variables for the sequence.
        
        Parameters
        ----------
        batch_size : int
            The size of the desired batch.
        dataset : str
            The name of the folder that contains the data. Note that the
            trailing slash should not be included e.g. "dataset", not
            "dataset/"
        cases: List[str]
            A list of all cases desired to be included in the sequence e.g.
            ["chb01", "chb02"]

        """
        self.batch_size = batch_size
        self.dataset = dataset
        self.cases = cases

        self.files = self._construct_file_list()
        self.test_file = self.files[0]
        self.files = self.files[1:]
        self.file_count = len(self.files)


    def _construct_file_list(self) -> List[str]:
        """Builds the list of filenames.
        
        Searches the dataset for all files matching cases, and cleans the
        names of them for easy use when getting the items (see __getitem__).

        Returns
        -------
        List[str]
            A list containing the filenames of files in the dataset matching
            the cases supplied. These filenames look like: `chb01_03.edf`.

        """
        # looks like chb01_03.edf_data.npy
        raw_filenames = [filename for filename in os.listdir(self.dataset) if os.path.basename(filename).split("_")[0] in self.cases]
        # looks like chb01_03.edf
        root_filenames = ["_".join(filename.split("_")[:-1]) for filename in raw_filenames]
        return root_filenames


    def __len__(self) -> int:
        return math.ceil(self.file_count / self.batch_size)
    

    def get_files(self) -> List[str]:
        return self.files
    

    def get_test_file(self) -> str:
        return self.test_file


    def __getitem__(self, idx) -> Tuple[NDArray, NDArray]:
        """Gets the current item, size of which is defined by batch size.
        
        Minimum granularity currently offered is whole files, in future this
        should be changed to be able to split files.

        Parameters
        ----------
        idx : int
            The index of the current item e.g. 0, 1, 2.

        Returns
        -------
        Tuple[Data, Labels]
            Data is the batch data, a list containing the data from the
            specified idx and batch size,
            Labels is the corresponding labels for that data.

        """
        low = idx * self.batch_size
        high = min(low + self.batch_size, self.file_count)

        first_run = True
        for i in range(low, high):
            data = np.load(self.dataset + "/" + self.files[i] + "_data.npy")
            labels = np.load(self.dataset + "/" + self.files[i] + "_labels.npy")

            if first_run:
                batch_data = data
                batch_labels = labels
                first_run = False
            else:
                batch_data = np.append(batch_data, data, axis=0)
                batch_labels = np.append(batch_labels, labels, axis=0)

        return batch_data, batch_labels
