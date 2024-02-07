DATA_ROOT = "dataset/chb-mit-scalp-eeg-database-1.0.0/"

from typing import Tuple, List
from re import findall
import mne
from mne.io.edf.edf import RawEDF
import numpy as np

class SeizureData:
    filename = ""
    seizure_count = 0
    # seizure number: [start, end]
    start_end = {}
    
    def __init__(self, filename: str, seizure_count: int, start_end: {int: Tuple[int, int]}):
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
    raw_file = mne.io.read_raw_edf(file)
    raw_file.crop(start_sec, end_sec)

    return raw_file


def generate_train_data(sequence_length: int, stride: int, cases: List[str]):
    for case in cases:
        case_seizure_data = get_seizure_data(case)
        for seizure_data in case_seizure_data:
            data_sequence = []
            labels = []

            raw = mne.io.read_raw_edf(seizure_data.filename, verbose=False)
            raw = raw.drop_channels("ECG", on_missing="ignore")
            data, timestamp = raw.get_data(return_times=True)

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
            # print(data_sequence)
            np.save("ml_processed/" + filename + "_data.npy", data_sequence)
            labels_reshaped = np.reshape(labels, (-1, 1)) # reshape necessary for keras model
            # print(labels_reshaped.shape)
            # print(labels[1])
            np.save("ml_processed/" + filename + "_labels.npy", labels_reshaped)


def generate_balanced_train_data(sequence_length: int, stride: int, cases: List[str]):
    """Currently only works with 1s sequence length and 256Hz stride (set both to 256)."""
    for case in cases:
        case_seizure_data = get_seizure_data(case)
        for seizure_data in case_seizure_data:
            if len(seizure_data.start_end) == 0:
                continue
            raw = mne.io.read_raw_edf(seizure_data.filename, verbose=False)
            ignore = ["ECG", "VNS", ".", "-", "-", "--0", "--1", "--2", "--3", "--4", "FC1-Ref", "FC2-Ref", "FC5-Ref", "FC6-Ref", "CP1-Ref", "CP2-Ref", "CP5-Ref", "CP6-Ref", "EKG1-CHIN"]
            raw = raw.drop_channels(ignore, on_missing="ignore")
            if len(raw.get_channel_types()) != 23:
                continue
            data, timestamp = raw.get_data(return_times=True)

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
            np.save("ml_processed_balanced/" + filename + "_data.npy", data_sequence)
            labels_reshaped = np.reshape(labels, (-1, 1)) # reshape necessary for keras model
            np.save("ml_processed_balanced/" + filename + "_labels.npy", labels_reshaped)


def generate_balanced_pre_ictal_train_data(sequence_length: int, stride: int, cases: List[str]):
    """Currently only works with 1s sequence length and 256Hz stride (set both to 256)."""
    for case in cases:
        case_seizure_data = get_seizure_data(case)
        for seizure_data in case_seizure_data:
            if len(seizure_data.start_end) == 0:
                continue
            raw = mne.io.read_raw_edf(seizure_data.filename, verbose=False)
            ignore = ["ECG", "VNS", ".", "-", "-", "--0", "--1", "--2", "--3", "--4", "FC1-Ref", "FC2-Ref", "FC5-Ref", "FC6-Ref", "CP1-Ref", "CP2-Ref", "CP5-Ref", "CP6-Ref", "EKG1-CHIN"]
            raw = raw.drop_channels(ignore, on_missing="ignore")
            if len(raw.get_channel_types()) != 23:
                continue
            data, timestamp = raw.get_data(return_times=True)

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
            np.save("ml_processed_balanced_pre_ictal/" + filename + "_data.npy", data_sequence)
            labels_reshaped = np.reshape(labels, (-1, 1)) # reshape necessary for keras model
            np.save("ml_processed_balanced_pre_ictal/" + filename + "_labels.npy", labels_reshaped)



def generate_balanced_pre_ictal_train_data(sequence_length: int, stride: int, cases: List[str], seconds_before: int):
    """Currently only works with 1s sequence length and 256Hz stride (set both to 256)."""
    for case in cases:
        case_seizure_data = get_seizure_data(case)
        for seizure_data in case_seizure_data:
            if len(seizure_data.start_end) == 0:
                continue
            raw = mne.io.read_raw_edf(seizure_data.filename, verbose=False)
            ignore = ["ECG", "VNS", ".", "-", "-", "--0", "--1", "--2", "--3", "--4", "FC1-Ref", "FC2-Ref", "FC5-Ref", "FC6-Ref", "CP1-Ref", "CP2-Ref", "CP5-Ref", "CP6-Ref", "EKG1-CHIN"]
            raw = raw.drop_channels(ignore, on_missing="ignore")
            if len(raw.get_channel_types()) != 23:
                continue
            data, timestamp = raw.get_data(return_times=True)

            negative_available = seconds_before * seizure_data.seizure_count
            data_sequence = []
            labels = []
            curr_seizure = 1
            for i in range(0, timestamp.shape[0] - sequence_length, stride):
                # if not in the x seconds before a seizure to the end of the seizure, label it 0 (not pre-ictal)
                if not seizure_data.start_end[curr_seizure][0] - seconds_before < timestamp[i] < seizure_data.start_end[curr_seizure][1] \
                    and negative_available > 0:
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
            np.save("ml_processed_balanced_pre_ictal/" + filename + "_pre_ictal_data.npy", data_sequence)
            labels_reshaped = np.reshape(labels, (-1, 1)) # reshape necessary for keras model
            np.save("ml_processed_balanced_pre_ictal/" + filename + "_pre_ictal_labels.npy", labels_reshaped)