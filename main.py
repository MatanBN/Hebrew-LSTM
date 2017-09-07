import numpy as np
import data_handler
import model

train_x, train_y = data_handler.read_data("train.txt", 50)
test_x, test_y = data_handler.read_data("test.txt", 50)

lstm = model.Model(y_shape=train_y.shape[2], batch_size=50)
weights_name = "weights_epoch_num10.h5"
print(weights_name)
lstm.load_weights(weights_name)
loss, acc = lstm.model.evaluate(test_x, test_y, verbose=0)
print("\nAccuaracy:", acc, "Loss:", loss)
generated_text = lstm.generate_some_text(200)
print(generated_text)
# lstm.train_model(train_x, train_y, test_x, test_y, epochs=100, save_weights_after=10)
