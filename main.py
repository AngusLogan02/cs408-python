from time import sleep
import mne
from data import get_seizure_data

DATA_ROOT = "dataset/chb-mit-scalp-eeg-database-1.0.0/"

chb12_data = get_seizure_data("chb01")
total = 0
for data in chb12_data:
    total += data.seizure_count
    # print(data.filename)

# print("filadmwakdm", chb12_data[0].filename)
# print(len(chb12_data))
print("total seizures in chb12", total)

# from data import get_seizure_data, get_time_window

# chb12 = get_seizure_data("chb12")
# first_seizure_start = chb12[0].start_end[1][0]
# first_seizure_end = chb12[0].start_end[1][1]

# print(chb12[0].filename, first_seizure_start, first_seizure_end)
# save_dir = "dataset/parsed/seizure/"

# first_seizure_edf = get_time_window(chb12[0].filename, first_seizure_start, first_seizure_end)
# first_seizure_edf.export(save_dir + chb12[0].filename.split('/')[-1])

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
