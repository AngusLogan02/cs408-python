DATA_ROOT = "dataset/chb-mit-scalp-eeg-database-1.0.0/"

from typing import Tuple, List
from re import findall
import mne
from mne.io.edf.edf import RawEDF

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
            print(file_name)
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