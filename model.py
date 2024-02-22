from keras.models import Sequential, Model
from keras.layers import Input, Conv1D, MaxPool1D, LSTM, Dense, Dropout
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

# TODO MAKE ALL QUOTES CONSISTENT

class ModelTools:
    def create_model(self, dropout=False) -> Model:
        """Create the CNN used for Ictal and Pre-Ictal detection.

        Parameters
        ----------
        dropout : bool
            Whether or not to enable the dropout layers

        Returns
        -------
        Keras Model
            A sequential model made up of CNN, (Dropout), and Dense layers.

        """
        model = Sequential()

        model.add(Input(23, 256))
        model.add(Conv1D(512, kernel_size=3, activation="relu"))
        model.add(MaxPool1D())
        if dropout:
            model.add(Dropout(0.25))
        model.add(Conv1D(256, kernel_size=3, activation="relu"))
        model.add(MaxPool1D())
        if dropout:
            model.add(Dropout(0.25))

        model.add(Conv1D(128, kernel_size=3, activation="relu"))
        model.add(MaxPool1D())
        if dropout:
            model.add(Dropout(0.25))

        model.add(Dense(256, activation="relu"))
        if dropout:
            model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))

        return model


    def compile_model(self, model) -> Model:
        """Compiles a specified model.

        Compiles the given model with the Adam optimiser and the binary
        cross-entropy loss function.
        
        Parameters
        ----------
        model : Keras Model

        Returns
        -------
        Compiled Keras Model
        
        """
        model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        return model


    def accuracy_loss_plot(self, history) -> None:
        """A utility function to simplify graphing a model's performance.

        Notes
        -----
        Could be improved to take in a list of histories, and iterate over
        them, combining them into one graph. This would enable the easy
        graphing of a model trained with multiple steps.
        
        Parameters
        ----------
        history : History
            The output of the model.fit method.

        Returns
        -------
        Nothing, but displays a line graph of the model's accuracy at each
        training epoch.

        """
        # Plotting accuracy and loss over epochs
        plt.figure(figsize=(12, 4))

        # Plotting accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"])
        plt.title("Model Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

        # Plotting loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"])
        plt.title("Model Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.tight_layout()
        plt.show()