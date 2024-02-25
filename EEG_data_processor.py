DATA_ROOT = "dataset/chb-mit-scalp-eeg-database-1.0.0/"

from typing import Dict, Tuple, List
from re import findall
import mne
from mne.io.edf.edf import RawEDF
import numpy as np
from numpy.typing import ArrayLike
from data import get_seizure_data

from labellers import Labeller

class EEG_Data_Processor:
    def get_clean_data(self, filename: str) -> Tuple[ArrayLike, ArrayLike]:
        """Drops non-EEG channels from a file and returns timestamped data in numpy
        form.
        
        Parameters
        ----------
        filename : str
            The path to the .edf file. 
        
        Returns
        -------
        Tuple[data, timestamp]
        OR
        Tuple[None, None]
            Returns None, None if the number of channels in the file is not 23.

        """
        raw = mne.io.read_raw_edf(filename, verbose=False)
        ignore = ["ECG", "VNS", ".", "-", "-", "--0", "--1", "--2", "--3", "--4", "FC1-Ref", "FC2-Ref", "FC5-Ref", "FC6-Ref", "CP1-Ref", "CP2-Ref", "CP5-Ref", "CP6-Ref", "EKG1-CHIN"]
        raw = raw.drop_channels(ignore, on_missing="ignore")
        if len(raw.get_channel_types()) != 23:
            return None, None
        data, timestamp = raw.get_data(return_times=True)
        return data, timestamp


    def save_labelled_data(self, folder: str, filename: str, data, labels, annotation: str = "") -> None:
        """Saves the data in .npy files using a specific naming convention.
        
        Parameters
        ----------
        folder : str
            The path to the folder to save the data in.
        filename : str
            The name of the file, e.g. "chb01-03.edf"
        data : ndarray
            The array containing the EEG data
        labels : ndarray
            The array containing the labels for the EEG data
        annotation : str, optional
            Text to add on to the end of the file name. It is recommended to 
            include an underscore at the start of the annotation for
            readability purposes.

        Returns
        -------
        Nothing, but saves the data and labels to the specified folder with correct
        naming.

        """
        np.save(folder + "/" + filename + annotation + "_data.npy", data)
        labels_reshaped = np.reshape(labels, (-1, 1)) # reshape necessary for keras model
        np.save(folder + "/" + filename + annotation + "_labels.npy", labels_reshaped)

    
    def generate_data(self, sequence_length, stride, cases: List[str], labeller: Labeller, output_dir: str):
        """Takes in EEG data in .edf form, labels it according to the labeller,
        then saves it in .npy form.
        
        Parameters
        ----------
        sequence_length : int
            The number of data points to include in one item, e.g. set to 256
            for 1 second of data in one item.
        stride : int
            The gap between each item, e.g. if both sequence_length and stride
            are set to 256, then each data point will contain 1 second of data
            with no overlap. If stride is set to 128 then the next item will
            start halfway through the last item.
        cases : List[str]
            The list of cases to generate data for, e.g. ["chb01", "chb02", ...].
        labeller : Labeller
            The labeller defines how the data is labelled. This may mean it will
            label a seizure as 1, a pre-ictal period as 1, or anything else. It
            is passed in so as to allow for flexibility in labelling.
        output_dir : str
            The output directory to save the labelled data in.

        Returns
        -------
        Nothing, but saves the labelled data in the specified output dir using
        `save_labelled_data`
        
        """
        for case in cases:
            case_seizure_data = get_seizure_data(case)
            for seizure_data in case_seizure_data:
                data_sequence = []
                labels = []

                if len(seizure_data.start_end) == 0:
                    continue
                data, timestamp = self.get_clean_data(seizure_data.filename)
                if data is None or timestamp is None:
                    continue

                for i in range(0, timestamp.shape[0] - sequence_length, stride):
                    label = labeller.label(seizure_data, timestamp[i])
                    if label is not None:
                        data_sequence.append(data[:, i:i+sequence_length])
                        labels.append(label)

                filename = seizure_data.filename.split("/")[-1]
                self.save_labelled_data(output_dir, filename, data_sequence, labels)
                labeller.reset()