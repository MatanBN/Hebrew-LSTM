from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Activation, Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt


class Model:
    def __init__(self, input_shape, y_shape, batch_size=1):
        # create and fit the model
        self.model = Sequential()
        self.model.add(LSTM(700, input_shape=(None, y_shape), return_sequences=True))
        self.model.add(Dropout(0.3, input_shape=(None, y_shape)))
        self.model.add(LSTM(700, return_sequences=True))
        self.model.add(LSTM(700, return_sequences=True))

        self.model.add(TimeDistributed(Dense(y_shape)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
        self.history = None
        self.batch_size = batch_size

    def train_model(self, train_x, train_y, val_x, val_y):
        for i in range(50):
            self.history = self.model.fit(train_x, train_y, epochs=1, batch_size=self.batch_size, verbose=1,
                                          shuffle=False,
                                          validation_data=(val_x, val_y))
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
