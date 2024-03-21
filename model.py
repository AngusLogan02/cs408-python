from keras.models import Sequential, Model
from keras.layers import Input, Conv1D, Conv2D, MaxPool1D, LSTM, Dense, Dropout
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

# TODO MAKE ALL QUOTES CONSISTENT

class ModelTools:
    def create_model(self, dropout=False, dropout_rate=0.25) -> Model:
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

        model.add(Input(shape=(23, 256)))
        model.add(Conv1D(512, kernel_size=5, activation="relu"))
        model.add(MaxPool1D())
        if dropout:
            model.add(Dropout(dropout_rate))
        model.add(Conv1D(256, kernel_size=3, activation="relu"))
        model.add(MaxPool1D())
        if dropout:
            model.add(Dropout(dropout_rate))

        model.add(Conv1D(128, kernel_size=1, activation="relu"))
        model.add(MaxPool1D())
        if dropout:
            model.add(Dropout(dropout_rate))

        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        return model

    def create_recurrent_model(self, dropout=False, dropout_rate=0.25) -> Model:
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

        model.add(Input(shape=(23, 256)))
        model.add(LSTM(512, activation="tanh"))
        if dropout:
            model.add(Dropout(dropout_rate))
        model.add(LSTM(256, activation="tanh"))
        if dropout:
            model.add(Dropout(dropout_rate))

        model.add(LSTM(128, activation="tanh"))
        if dropout:
            model.add(Dropout(dropout_rate))

        model.add(Dense(128, activation="sigmoid"))
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


    def accuracy_loss_plot(self, histories, filename=None) -> None:
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
        combined_history = {}
        for history in histories:
            for key in history.history.keys():
                if combined_history.get(key) is not None:
                    combined_history[key] = combined_history[key] + history.history[key]
                else:
                    combined_history[key] = history.history[key]
        # Plotting accuracy and loss over epochs
        plt.figure(figsize=(12, 4))

        # Plotting accuracy
        plt.subplot(1, 2, 1)
        plt.plot(combined_history["accuracy"])
        plt.title("Model Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

        # Plotting loss
        plt.subplot(1, 2, 2)
        plt.plot(combined_history["loss"])
        plt.title("Model Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()