from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Activation
from keras.layers import Dense, TimeDistributed, Activation, Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt
import numpy as np


class Model:
    def __init__(self, input_shape, y_shape, batch_size=1):
        # create and fit the model
        self.model = Sequential()
        self.model.add(
            LSTM(y_shape, input_shape=(None, y_shape), return_sequences=True))
        self.model.add(Dropout(0.20))

        self.model.add(LSTM(512, return_sequences=True))
        self.model.add(Dropout(0.20))

        self.model.add(LSTM(512, return_sequences=True))
        self.model.add(Dropout(0.20))

        self.model.add(LSTM(256, return_sequences=True))
        self.model.add(Dropout(0.20))

        self.model.add(Dense(y_shape))
        self.model.add(Activation('softmax'))

        # compile or load weights then compile depending
        self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
        self.history = None
        self.batch_size = batch_size

    def test_model(self, test_x, test_y):
        cross_entropy = 0
        prediction_accuracy = 0
        test_length = len(test_x)
        correct_predictions_counter = 0
        predictions = self.model.predict(test_x)

        for i in range(test_length):
            prediction = predictions[i]
            predicted_char_index = np.argmax(prediction)
            actual_char_index = np.argmax(test_y[i])
            cross_entropy = ((cross_entropy * i) - np.log2(prediction[0][actual_char_index])) / (i + 1)
            if (test_y[i][0][predicted_char_index] == 1):
                correct_predictions_counter += 1

        prediction_accuracy = correct_predictions_counter / test_length

        return prediction_accuracy, cross_entropy

    def load_weights(self, file_path):
        self.model.load_weights(filepath=file_path)

    def train_model(self, train_x, train_y, val_x, val_y):
        for i in range(50):
            print("Epoch Num: ", str(i + 1))
            self.history = self.model.fit(train_x, train_y, epochs=1, batch_size=self.batch_size, verbose=1,
                                          shuffle=False)
            self.model.reset_states()

            pred_acc, cros_entrop = self.test_model(val_x, val_y)
            print("Validation Accuracy: ", pred_acc, " Validation Cross Entropy: ", cros_entrop)
            self.model.reset_states()

        self.model.save(filepath="weights.h5")

    def plot_results(self):
        if self.history is not None:
            plt.plot(self.history.history['acc'])
            plt.plot(self.history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
        else:
            print("No training were done yet")

    def generate_some_text(self, amount_of_chars, first_chars=[]):
        result = []

        self.model.predict(first_chars)
        last_char = first_chars[-1]
        one_hot_len = len(last_char)

        for i in range(amount_of_chars):
            prediction = self.model.predict(last_char)
            predicted_char_index = np.argmax(prediction)
            last_char = np_utils.to_categorical(np.array([predicted_char_index]), one_hot_len)
            result.append(usable_chars[predicted_char_index])

        return "".join(result)
