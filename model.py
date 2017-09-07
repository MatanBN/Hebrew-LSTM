from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Activation, regularizers
from keras.layers import Dense, TimeDistributed, Activation, Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from data_handler import usable_chars
from data_handler import add_suffixes

from keras.callbacks import Callback


class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.N == 0:
            name = 'weights_epoch_num' + str(epoch) + '.h5'
            self.model.save_weights(name)
        self.batch += 1


# Model class contains the lstm model obejcts and the methods that the model is responsible for.
class Model:
    # Constructor initializes the model.
    def __init__(self, y_shape, batch_size=1):
        # create and fit the model
        self.model = Sequential()
        self.model.add(
            LSTM(128, input_shape=(None, y_shape), return_sequences=True, implementation=2))
        self.model.add(TimeDistributed(Dense(y_shape)))
        self.model.add(Activation('softmax'))

        # compile or load weights then compile depending
        self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
        self.history = None
        self.batch_size = batch_size

    # Tests the model according to the test set given.
    def test_model(self, test_x, test_y):
        cross_entropy = 0
        prediction_accuracy = 0
        test_length = len(test_x)
        predictions = self.model.predict(test_x)

        for i in range(test_length):
            prediction = predictions[i]
            for j in range(prediction.shape[0]):
                predicted_char_index = np.argmax(prediction[j])
                actual_char_index = np.argmax(test_y[i][j])
                cross_entropy -= np.log2(prediction[j][actual_char_index])
                if test_y[i][j][predicted_char_index] == 1:
                    prediction_accuracy += 1

        prediction_accuracy = prediction_accuracy / test_length
        cross_entropy /= test_length

        return prediction_accuracy, cross_entropy

    # Saves the current weights to the file path given.
    def save_weights(self, file_path):
        self.model.save_weights(filepath=file_path, overwrite=True)

    # Loads the weights from the file path given.
    def load_weights(self, file_path):
        self.model.load_weights(filepath=file_path)

    # Trains the model with the train_x and train_y data, the number of epochs is determined according to epochs variable, and the
    def train_model(self, train_x, train_y, val_x, val_y, epochs=1, save_weights_after=10):
        self.history = self.model.fit(train_x, train_y, epochs=epochs, batch_size=self.batch_size,
                                      verbose=1, shuffle=False, validation_data=(val_x, val_y),
                                      callbacks=[WeightsSaver(self.model, save_weights_after)])

    # Plots the result of the training.
    def plot_results(self):
        if self.history is not None:
            plt.plot(self.history.history['acc'])
            plt.title('training accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()
        else:
            print("No training were done yet")

    """
    Generates a new text in the length of amount_of_chars with the help of first_chars or with a random first letter
    when no characters are given.
    """

    def generate_some_text(self, amount_of_chars, first_chars=np.array([])):
        ix = [np.random.randint(len(usable_chars))]

        y_char = []
        X = np.zeros((1, amount_of_chars, len(usable_chars)))
        for i in range(0, first_chars.shape[0]):
            for j in range(first_chars[i][0].size):
                X[0][i][j] = first_chars[i][0][j]
            ix = [np.argmax(first_chars[i][0])]
            y_char.append(usable_chars[ix[-1]])
        start_from = max(0, first_chars.shape[0] - 1)
        for i in range(start_from, amount_of_chars):
            X[0, i, :][ix[-1]] = 1
            # print(usable_chars[ix[-1]], end="")
            prediction = self.model.predict(X[:, :i + 1, :])[0][i]

            ix = np.random.choice(len(usable_chars), 1, p=prediction)

            # ix = np.argmax(self.model.predict(X[:, :i + 1, :])[0], 1)
            y_char.append(usable_chars[ix[-1]])

        generated_text = ('').join(y_char)
        return add_suffixes(generated_text)
