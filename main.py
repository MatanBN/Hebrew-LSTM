import sys

from keras.callbacks import ModelCheckpoint

import data_handler
import model
from data_handler import usable_chars

train_x, train_y = data_handler.read_data("train.txt", 50)
test_x, test_y = data_handler.read_data("test.txt", 50)
lstm = model.Model(y_shape=len(usable_chars), batch_size=50)
weights_file = "weights.hdf5"
checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
if len(sys.argv) > 1:
    lstm.load_weights(sys.argv[1])
else:
    lstm.train_model(train_x, train_y, test_x, test_y, epochs=50, callbacks=callbacks_list)
    lstm.load_weights("weights.hdf5")

acc, loss = lstm.test_model(train_x, train_y)
print("Training Accuaracy:", acc, "Training Loss:", loss)
acc, loss = lstm.test_model(test_x, test_y)
print("Test Accuaracy:", acc, "Test Loss:", loss)
generated_text = lstm.generate_some_text(10000)
file = open("201120441_317867760.txt", 'w')
file.write(generated_text)