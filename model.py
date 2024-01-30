from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPool1D, LSTM, Dense, Dropout
import matplotlib.pyplot as plt


def create_model(input_dim=(23, 256)):
    model = Sequential()

    model.add(Input(input_dim))
    model.add(Conv1D(512, kernel_size=3, activation="relu"))
    model.add(MaxPool1D())
    model.add(Conv1D(256, kernel_size=3, activation="relu"))
    model.add(MaxPool1D())
    model.add(Conv1D(128, kernel_size=3, activation="relu"))
    model.add(MaxPool1D())
    # model.add(LSTM(64, activation="tanh"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    return model


def create_recurrent_model():
    model = Sequential()

    model.add(LSTM(256, activation="tanh", return_sequences=True, input_shape=(23, 7)))
    model.add(LSTM(512, activation="tanh", return_sequences=True))
    model.add(LSTM(256, activation="tanh"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(1, activation="sigmoid"))

    return model

def compile_model(model):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def accuracy_loss_plot(history):
    # Plotting accuracy and loss over epochs
    plt.figure(figsize=(12, 4))

    # Plotting accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()