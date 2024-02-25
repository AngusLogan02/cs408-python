import os
import numpy as np
from seizure_sequence import SeizureSequence
from model import create_model, create_large_model, compile_model

# TODO remove unused imports, commented lines, types to variables
cases = [x for x in os.listdir("dataset/chb-mit-scalp-eeg-database-1.0.0/") if "chb" in x]
cases = cases[16:]
print(cases)
for case in cases:
    if case == "chb04" or case == "chb07" or case == "chb15" or case == "chb17" or case == "chb18"\
        or case == "chb20":
        continue
    print("//////////////////////////", case)
    seizure_sequence_balanced = SeizureSequence(1, "ml_processed_balanced_pre_ictal", case, bias_positive=False)
    
    model = compile_model(create_model())

    history = model.fit(seizure_sequence_balanced, epochs=200)

    file = seizure_sequence_balanced.get_test_file()

    whole_file_data = np.load("ml_processed_pre_ictal/" + file + "_data.npy")
    whole_file_labels = np.load("ml_processed_pre_ictal/" + file + "_labels.npy")
    # whole_file_labels = np.load("ml_processed/chb04_08.edf_labels.npy")
    # whole_file_labels = np.load("ml_processed/chb04_08.edf_labels.npy")

    print(whole_file_data.shape)
    predictions = model.predict(whole_file_data)

    threshold = 0.9
    best_threshold = threshold
    best_accuracy = 0
    lowest_false_neg = float('inf')
    detail = 10
    for i in range(0, 101):
        false_pos = 0
        true_pos = 0
        false_neg = 0
        true_neg = 0
        for i, prediction in enumerate(predictions):
            p = prediction[0][0]
            # if p == 1:
            if p > threshold:
                if whole_file_labels[i] == 0:
                    false_pos += 1
                else:
                    true_pos += 1
            else:
                if whole_file_labels[i] == 1:
                    false_neg += 1
                else:
                    true_neg += 1
        whole_file_accuracy = ((true_pos + true_neg) / whole_file_data.shape[0]) * 100
        if false_neg <= lowest_false_neg and true_pos > 0 and whole_file_accuracy > best_accuracy:
            lowest_false_neg = false_neg
            best_accuracy = whole_file_accuracy
            best_threshold = threshold
        threshold = round(threshold + 0.001, 3)

    # print(f"Best threshold: {best_threshold}\nBest accuracy: {best_accuracy} \nLowest false neg: {lowest_false_neg}")



    false_pos = 0
    true_pos = 0
    false_neg = 0
    true_neg = 0
    for i, prediction in enumerate(predictions):
        p = prediction[0][0]
        # if p == 1:
        if p > best_threshold:
            if whole_file_labels[i] == 0:
                false_pos += 1
            else:
                true_pos += 1
        else:
            if whole_file_labels[i] == 1:
                false_neg += 1
            else:
                true_neg += 1
    whole_file_accuracy = ((true_pos + true_neg) / whole_file_data.shape[0]) * 100
    print("\t false", "\t", "true")
    print("pos \t", false_pos, "\t", true_pos)
    print("neg \t", false_neg, "\t", true_neg)
    print(f"\nBest Threshold: {best_threshold}, Correct: {round(whole_file_accuracy, 2)}%")

    if True and false_neg == 0:
        model.save("models/" + case + "_threshold_" + str(best_threshold) + "_accuracy_" + str(round(best_accuracy)) + ".h5")