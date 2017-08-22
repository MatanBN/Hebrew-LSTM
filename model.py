from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt

class Model:

    def __init__(self, input_shape, y_shape):
        # create and fit the model
        self.model = Sequential()
        self.model.add(LSTM(32, input_shape=input_shape))
        self.model.add(Dense(y_shape, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history = None

    def train_model(self, train_x, train_y):
        self.history = self.model.fit(train_x, train_y, epochs=50, batch_size=1, verbose=1)

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


