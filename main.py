import data_extractor
import model

train_x, train_y, val_x, val_y, test_x, test_y = data_extractor.read_data("data.txt", 100000, 10000)
lstm = model.Model(input_shape=(train_x.shape[0], train_x.shape[1]), y_shape=train_y.shape[2], batch_size=64)
# lstm.train_model(train_x, train_y, val_x, val_y)
lstm.load_weights('weights.h5')
prediction_accuracy, cross_entropy = lstm.test_model(test_x, test_y)
print("Prediction accuracy: {a}% \n"
      "Cross entropy: {b}"
      .format(a=prediction_accuracy, b=cross_entropy))
lstm.plot_results()
