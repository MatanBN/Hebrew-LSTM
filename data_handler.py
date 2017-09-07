import numpy as np

usable_chars = [' ', 'ו', 'י', 'ה', 'מ', 'ל', 'א', 'ר', 'ב', 'נ', 'ת', 'ש', 'ע', 'כ', ',', 'ד', '.', 'ח', 'פ', 'ק', '-',
                'צ', 'ג', 'ס', 'ז', '"', 'ט', '?', '!', ':', '\'', '\n']

punctuations = [' ', ',', '.', '-', '"', '?', '!', ':', '\'', '\n']

letter_to_finals = {u'מ': u'ם', u'נ': u'ן', u'פ': u'ף', u'צ': u'ץ', u'כ': u'ך'}

# Converts a text file to numbers according to the usable_chars mapping
def letters_to_numbers(letters):
    data = []
    for letter in letters:
        data.append(usable_chars.index(letter))
    return np.asarray(data)


# Formats the data set to the one hot array for the input and output
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


# Removes the suffixes chars in the text
def remove_suffixes(text):
    text = text.replace(u'ף', 'פ')
    text = text.replace(u'ם', 'מ')
    text = text.replace(u'ץ', 'צ')
    text = text.replace(u'ן', 'נ')
    text = text.replace(u'ך', 'כ')

    return text

# Create the dataset in the right format for lstm.
def create_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    dataset_len = int(len(dataset) / look_back)
    for i in range(dataset_len):
        a = dataset[i * look_back:(i + 1) * look_back]
        b = dataset[i * look_back + 1:(i + 1) * look_back + 1]
        data_x.append(a)
        data_y.append(b)
    data_x, data_y = format_dataset(data_x, data_y, seq_length=look_back)
    return data_x, data_y

    # Add the finals to the text.
def add_suffixes(string):
    new_string = []
    string_length = len(string) - 1
    for i in range(string_length):
        if string[i] in letter_to_finals.keys() and string[i + 1] in punctuations:
            new_string.append(letter_to_finals[string[i]])
        else:
            new_string.append(string[i])
    if string[i] in letter_to_finals.keys():
        new_string.append(letter_to_finals[string[i]])
    else:
        new_string.append(string[i])

    return ('').join(new_string)

"""
Get a txt file name and returns a tuple of np arrays of x_data for train, validation and test and y_data for train,
validation and test which will be in the format of one shot np arrays according to the usable_chars mapping.
"""


def read_data(file_name, sequence=1):
    file = open(file_name, 'r', encoding='utf-8')
    text = file.read()

    text = remove_suffixes(text)

    text = letters_to_numbers(text)

    (x, y) = create_dataset(text, sequence)
    return (x, y)
