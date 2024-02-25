from time import sleep
import mne
from data import get_seizure_data

DATA_ROOT = "dataset/chb-mit-scalp-eeg-database-1.0.0/"

from data import get_seizure_data, get_time_window

edf_data = get_seizure_data("chb01")

for i, file in enumerate(edf_data):
    for seizure_number, start_end in file.start_end.items():
        seizure_start = start_end[0]
        seizure_end = start_end[1]

        save_dir = "dataset/parsed/seizure/"
        filename = file.filename.split('/')[-1].split('.')
        export_filename = save_dir + filename[0] + '_' + str(seizure_number) + '.' + filename[1]

        seizure_edf = get_time_window(file.filename, seizure_start, seizure_end)

        seizure_edf.export(export_filename, verbose=False, overwrite=True)

# quit()
# raw_edf = mne.io.read_raw_edf(DATA_ROOT + "chb12/chb12_27.edf")
# # raw_edf_seizure = mne.io.read_raw_edf("dataset/chb01_03_seizures.edf")

# raw_edf.load_data()
# # raw_edf.plot()
# print("edf plotted")
# raw_edf.filter(l_freq=40, h_freq=None)
# raw_edf.plot()
# # raw_edf_seizure.load_data()
# # raw_edf_seizure.filter(l_freq=40, h_freq=None)
# # raw_edf_seizure.plot()
# input()
