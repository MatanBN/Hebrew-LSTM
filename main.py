import data_extractor
import model
import numpy as np

train_x, train_y, test_x, test_y = data_extractor.read_data("data-train.txt", 100000, 1)
lstm = model.Model(input_shape=(train_x.shape[0], train_x.shape[1]), y_shape=train_y.shape[2], batch_size=100)
# lstm.train_model(train_x, train_y, test_x, test_y)
lstm.load_weights('weights_epoch_num80.h5')
lstm.plot_results()
generated_text = lstm.generate_some_text(200, np.array(test_x[0:100]))
print(generated_text)
