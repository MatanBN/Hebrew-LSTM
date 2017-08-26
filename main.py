import data_extractor
import model

train_x, train_y, test_x, test_y = data_extractor.read_data("brenner-train.txt", 100000, sequence=50)

lstm = model.Model(input_shape=(train_x.shape[0], train_x.shape[1]), y_shape=train_y.shape[2], batch_size=64)
lstm.train_model(train_x, train_y)
lstm.plot_results()
