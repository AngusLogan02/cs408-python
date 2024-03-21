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