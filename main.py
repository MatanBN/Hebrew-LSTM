import data_extractor
import model

train_x, train_y, test_x, test_y = data_extractor.read_data("brenner-train.txt", 10000, sequence=3)

lstm = model.Model(input_shape=(train_x.shape[1], train_x.shape[2]), y_shape=train_y.shape[1])
lstm.train_model(train_x, train_y)
lstm.plot_results()
