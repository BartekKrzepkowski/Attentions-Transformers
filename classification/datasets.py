import numpy as np
from torch.utils.data import Dataset
from tensorflow.keras.datasets import imdb


class IMDBTrain(Dataset):
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = imdb.load_data(
            path='imdb.npz',
            num_words=10000,
            skip_top=10,
            maxlen=40,
            seed=113,
            start_char=1,
            oov_char=2,
            index_from=3
        )
        self.x_train = x_train
        self.y_train = y_train
        self.word2index = imdb.get_word_index()

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, index):
        '''
            __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalize source and
            target values using the vocabulary objects we created in __init__
        '''
        x, y = self.x_train[index], self.y_train[index]
        return x, y


class IMDBTest(Dataset):
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = imdb.load_data(
            path='imdb.npz',
            num_words=10000,
            skip_top=10,
            maxlen=40,
            seed=113,
            start_char=1,
            oov_char=2,
            index_from=3
        )
        self.x_test = x_test
        self.y_test = y_test
        self.word2index = imdb.get_word_index()

    def __len__(self):
        return len(self.y_test)

    def __getitem__(self, index):
        '''
            __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalize source and
            target values using the vocabulary objects we created in __init__
        '''
        x, y = self.x_test[index], self.y_test[index]
        return x, y