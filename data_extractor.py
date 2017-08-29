import numpy as np

usable_chars = [' ', 'ו', 'י', 'ה', 'מ', 'ל', 'א', 'ר', 'ב', 'נ', 'ת', 'ש', 'ע', 'כ', ',', 'ד', '.', 'ח', 'פ', 'ק',
                '-', 'צ', 'ג', 'ס', 'ז', '"', 'ט', '?', '!', ':', '\'', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '0']


# Converts a text file to numbers according to the usable_chars mapping
def letters_to_numbers(letters):
    data = []
    for letter in letters:
        data.append(usable_chars.index(letter))
    return np.asarray(data)


def format_dataset(data_x, data_y, seq_length=1, max_len=1):
    x = np.zeros((len(data_x), seq_length, len(usable_chars)))
    for i in range(len(data_x)):
        for j in range(len(data_x[i])):
            x[i][j][data_x[i][j]] = 1.0
    # x = np.array(x)
    # x = pad_sequences(data_x, maxlen=max_len, dtype='float32')
    # reshape X to be [samples, time steps, features]
    # x = np.reshape(x, (x.shape[0], max_len, 1))
    # normalize
    # x = x / float(len(usable_chars))
    y = np.zeros((len(data_x), seq_length, len(usable_chars)))
    for i in range(len(data_y)):
        for j in range(len(data_y[i])):
            y[i][j][data_y[i][j]] = 1.0
    return x, y


# Create the dataset in the right format for lstm.
def create_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    dataset_len = int(len(dataset) / look_back)
    for i in range(dataset_len):
        a = dataset[i * look_back:(i + 1) * look_back]
        b = dataset[i * look_back + 1:(i + 1) * look_back + 1]
        data_x.append(a)
        data_y.append(dataset[b])
    data_x, data_y = format_dataset(data_x, data_y, seq_length=look_back)
    return data_x, data_y


"""
Get a txt file name and returns a tuple of np arrays of x_data for train, validation and test and y_data for train,
validation and test which will be in the format of one shot np arrays according to the usable_chars mapping.
"""


def read_data(file_name, test_size, validation_size, sequence=1):
    file = open(file_name, 'r', encoding='utf-8')
    text = file.readline()
    train_size = len(text) - test_size - validation_size

    train_text = text[:train_size]
    val_text = text[train_size: train_size + validation_size]
    test_text = text[train_size + validation_size:]

    train_x = letters_to_numbers(train_text)
    val_x = letters_to_numbers(val_text)
    test_x = letters_to_numbers(test_text)

    (train_x, train_y) = create_dataset(train_x, sequence)
    (val_x, val_y) = create_dataset(val_x, sequence)
    (test_x, test_y) = create_dataset(test_x, look_back=1)
    return (train_x, train_y, val_x, val_y, test_x, test_y)
