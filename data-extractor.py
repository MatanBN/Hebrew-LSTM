import numpy as np
usable_chars = [' ', 'ו', 'י', 'ה', 'מ', 'ל', 'א', 'ר', 'ב', 'נ', 'ת', 'ש', 'ע', 'כ', ',', 'ד', '.', 'ח', 'פ', 'ק',
                '-', 'צ', 'ג', 'ס', 'ז', '"', 'ט', '?', '!', ':', '\'', '1','2','3','4','5','6','7','8','9','0']

# Convert a letter to a one hot np array according to the usable_chars mapping.
def letter_to_vec(letter):
    letter_vec = np.zeros(len(usable_chars))
    letter_vec[usable_chars.index(letter)] = 1
    return letter_vec

def letters_to_matrix(letters):
    data = []
    for letter in letters:
        data.append(letter_to_vec(letter))
    return np.asarray(data)

# Create the right predictions of each letter in the dataset.
def get_predictions(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# Get a txt file name and returns a tuple of np arrays of one shot np arrays according to the usable_chars mapping.
def read_data(file_name, test_size):
    file = open(file_name, 'r', encoding='utf-8')
    text = file.readline()
    train_size = len(text) - test_size

    train_text = text[:train_size]
    test_text = text[train_size:]

    train_data = letters_to_matrix(train_text)
    test_data = letters_to_matrix(test_text)

    return (train_data, test_data)





