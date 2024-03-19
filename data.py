DATA_ROOT = "dataset/chb-mit-scalp-eeg-database-1.0.0/"

from typing import Dict, Tuple, List
from re import findall
import mne
from mne.io.edf.edf import RawEDF
import numpy as np
from numpy.typing import ArrayLike


class SeizureData:
    """A data class to make parsing and working with the dataset easier.

    Contains information about a .edf file in the dataset.

    """
    
    def __init__(self, filename: str, seizure_count: int, start_end: Dict[int, Tuple[int, int]]):
        """
        Parameters
        ----------
        filename : str
            The path of the .edf file containing the EEG data.
        seizure_count : int
            The total number of seizures in the file.
        start_end : Dict[int, Tuple[int, int]]
            A dictionary mapping the seizure number to a tuple containing the
            start and end seconds of the seizure.
        
        """
        self.filename = filename
        self.seizure_count = seizure_count
        self.start_end = start_end

    def __str__(self) -> str:
        start_end_string = ""
        for seizure_number, start_end in self.start_end.items():
            start_end_string = start_end_string + \
                f"\nSeizure {seizure_number} starts at {start_end[0]} seconds, and ends at {start_end[1]} seconds."
        return f"File {self.filename} contains {self.seizure_count} seizures.{start_end_string}" 

def get_seizure_data(case: str) -> List[SeizureData]:
    """Parses information about a case.
    
    Parameters
    ----------
    case : str
        The name of the case e.g. "chb01"

    Returns
    -------
    List[SeizureData]
        A list containing a SeizureData object for each .edf file in the case.
    
    """
    # TODO split this into diff functions
    file = open(DATA_ROOT + case + "/" + case + "-summary.txt", mode="r")
    lines = file.readlines()
    file.close()

    all_file_data = []
    file_name = None
    for i in range(0, len(lines)):
        if lines[i].startswith("File Name"):
            file_name = DATA_ROOT + case + "/" + [word for word in lines[i].split() if word.endswith(".edf")][0]
            starts = {}
            ends = {}
            seizure_count = 0

        if lines[i].startswith("Number of"):
            seizure_count = int(lines[i][-2])
        # TODO make this and next if one func
        if lines[i].startswith("Seizure") and "Start" in lines[i]:
            numbers_in_string = [int(n) for n in findall(r"\d+", lines[i])]
            if len(numbers_in_string) == 1:
                starts[1] = numbers_in_string[0]
            else:
                starts[numbers_in_string[0]] = numbers_in_string[1]

        if lines[i].startswith("Seizure") and "End" in lines[i]:
            numbers_in_string = [int(n) for n in findall(r"\d+", lines[i])]
            if len(numbers_in_string) == 1:
                ends[1] = numbers_in_string[0]
            else:
                ends[numbers_in_string[0]] = numbers_in_string[1]

        if file_name is not None and len(lines[i].split()) == 0 or i == i-1:
            start_end = {}
            for seizure_number in starts.keys():
                start_end[seizure_number] = (starts[seizure_number], ends[seizure_number])
            seizure_data = SeizureData(file_name, seizure_count, start_end)
            all_file_data.append(seizure_data)
            file_name = None
    
    return all_file_data

def get_time_window(file: str, start_sec: int, end_sec: int) -> RawEDF:
    """Crops the specified window from a .edf file.
    
    Parameters
    ----------
    file : str
        The path of the .edf file to extract the window from.
    start_sec : int
        The start second of the window to crop.
    end_sec : int
        The end second of the window to crop.

    Returns
    -------
    RawEDF
        A RawEDF object containing the desired window.

    """
    raw_file = mne.io.read_raw_edf(file)
    raw_file.crop(start_sec, end_sec)

    return raw_file

# TODO type hints
# TODO make these util functions under a class e.g. EEG_PROCESSOR

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
            Text to add on to the end of the file name

        Returns
        -------
        Nothing, but saves the data and labels to the specified folder with correct
        naming.

        """
        np.save(folder + "/" + filename + "_" + annotation + "_data.npy", data)
        labels_reshaped = np.reshape(labels, (-1, 1)) # reshape necessary for keras model
        np.save(folder + "/" + filename + "_" + annotation + "_labels.npy", labels_reshaped)

    
    def generate_data(self, cases: List[str], labeller: any):
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


    def generate_train_data(sequence_length: int, stride: int, cases: List[str]):
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

                seizure_count = 1
                for i in range(0, timestamp.shape[0] - sequence_length, stride):
                    data_sequence.append(data[:, i:i+sequence_length])
                    label = 0
                    if len(seizure_data.start_end) + 1 > seizure_count:
                        if seizure_data.start_end[seizure_count][0] < timestamp[i] < seizure_data.start_end[seizure_count][1]:
                            label = 1
                        if timestamp[i] > seizure_data.start_end[seizure_count][1]:
                            seizure_count += 1
                    labels.append(label)

                filename = seizure_data.filename.split("/")[-1]
                self.save_labelled_data("refactor_test", filename, data_sequence, labels)


    def generate_balanced_train_data(sequence_length: int, stride: int, cases: List[str]):
        """Currently only works with 1s sequence length and 256Hz stride (set both to 256)."""
        for case in cases:
            case_seizure_data = get_seizure_data(case)
            for seizure_data in case_seizure_data:
                if len(seizure_data.start_end) == 0:
                    continue
                data, timestamp = self.get_clean_data(seizure_data.filename)
                if data is None or timestamp is None:
                    continue

                seconds_in_file = 0 # used to determine if more negative data can be added
                for seizure in seizure_data.start_end.values():
                    seconds_in_file += seizure[1] - seizure[0]
                data_sequence = []
                labels = []
                curr_seizure = 1
                for i in range(0, timestamp.shape[0] - sequence_length, stride):
                    if not seizure_data.start_end[curr_seizure][0] < timestamp[i] < seizure_data.start_end[curr_seizure][1] \
                        and seconds_in_file > 0:
                        data_sequence.append(data[:, i:i+sequence_length])
                        labels.append(0)
                        seconds_in_file -= 1
                    elif seizure_data.start_end[curr_seizure][0] <= timestamp[i] < seizure_data.start_end[curr_seizure][1]:
                        data_sequence.append(data[:, i:i+sequence_length])
                        labels.append(1)
                    
                    if timestamp[i] > seizure_data.start_end[curr_seizure][1] and curr_seizure < len(seizure_data.start_end):
                        curr_seizure += 1
                
                filename = seizure_data.filename.split("/")[-1]
                save_labelled_data("refactor_test", filename, data_sequence, labels)


    def generate_balanced_pre_ictal_train_data(self, sequence_length: int, stride: int, cases: List[str], seconds_before: int, output_dir: str):
        """Currently only works with 1s sequence length and 256Hz stride (set both to 256)."""
        for case in cases:
            print("ANJSDNJ")
            case_seizure_data = get_seizure_data(case)
            for seizure_data in case_seizure_data:
                if len(seizure_data.start_end) == 0:
                    continue
                data, timestamp = self.get_clean_data(seizure_data.filename)
                if data is None or timestamp is None:
                    continue

                negative_available = seconds_before * seizure_data.seizure_count
                data_sequence = []
                labels = []
                curr_seizure = 1
                for i in range(0, timestamp.shape[0] - sequence_length, stride):
                    # if not in the x seconds before a seizure to the end of the seizure, label it 0 (not pre-ictal)
                    if not seizure_data.start_end[curr_seizure][0] - seconds_before < timestamp[i] < seizure_data.start_end[curr_seizure][1] \
                        and negative_available > 0:
                        print(data.shape)
                        data_sequence.append(data[:, i:i+sequence_length])
                        labels.append(0)
                        negative_available -= 1
                    # else if in the x seconds before a seizure, label it 1 (pre-ictal)
                    elif seizure_data.start_end[curr_seizure][0] - seconds_before <= timestamp[i] < seizure_data.start_end[curr_seizure][0]:
                        data_sequence.append(data[:, i:i+sequence_length])
                        labels.append(1)
                    
                    if timestamp[i] > seizure_data.start_end[curr_seizure][1] - seconds_before and curr_seizure < len(seizure_data.start_end):
                        curr_seizure += 1
                
                filename = seizure_data.filename.split("/")[-1]
                # self.save_labelled_data("refactor_test", filename, data_sequence, labels, annotation="test")

    def generate_pre_ictal_train_data(sequence_length: int, stride: int, cases: List[str], seconds_before: int):
        """Currently only works with 1s sequence length and 256Hz stride (set both to 256)."""
        for case in cases:
            print("ahahahha")
            case_seizure_data = get_seizure_data(case)
            for seizure_data in case_seizure_data:
                if len(seizure_data.start_end) == 0:
                    continue
                raw = mne.io.read_raw_edf(seizure_data.filename, verbose=False)
                ignore = ["ECG", "VNS", ".", "-", "-", "--0", "--1", "--2", "--3", "--4", "FC1-Ref", "FC2-Ref", "FC5-Ref", "FC6-Ref", "CP1-Ref", "CP2-Ref", "CP5-Ref", "CP6-Ref", "EKG1-CHIN"]
                raw = raw.drop_channels(ignore, on_missing="ignore")
                if len(raw.get_channel_types()) != 23:
                    print(seizure_data)
                    continue
                data, timestamp = raw.get_data(return_times=True)

                data_sequence = []
                labels = []
                curr_seizure = 1
                for i in range(0, timestamp.shape[0] - sequence_length, stride):
                    # if not in the x seconds before a seizure to the end of the seizure, label it 0 (not pre-ictal)
                    if not seizure_data.start_end[curr_seizure][0] - seconds_before < timestamp[i] < seizure_data.start_end[curr_seizure][1]:
                        data_sequence.append(data[:, i:i+sequence_length])
                        labels.append(0)
                    # else if in the x seconds before a seizure, label it 1 (pre-ictal)
                    elif seizure_data.start_end[curr_seizure][0] - seconds_before <= timestamp[i] < seizure_data.start_end[curr_seizure][0]:
                        data_sequence.append(data[:, i:i+sequence_length])
                        labels.append(1)
                    
                    if timestamp[i] > seizure_data.start_end[curr_seizure][1] - seconds_before and curr_seizure < len(seizure_data.start_end):
                        curr_seizure += 1
                
                filename = seizure_data.filename.split("/")[-1]
                save_labelled_data("refactor_test", filename, data_sequence, labels, annotation="test")