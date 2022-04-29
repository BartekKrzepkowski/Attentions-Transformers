import numpy as np
from torch.utils.data import Dataset


class LM_Dataset(Dataset):
    '''
    Initiating Variables
    df: the training dataframe
    source_column : the name of source text column in the dataframe
    target_columns : the name of target text column in the dataframe
    transform : If we want to add any augmentation
    freq_threshold : the minimum times a word must occur in corpus to be treated in vocab
    source_vocab_max_size : max source vocab size
    target_vocab_max_size : max target vocab size
    '''

    def __init__(self, path, pad_token='<pad>', min_sent_lenght=4, max_sent_lenght=40):

        self.Ind2word = None
        self.pad_token = pad_token
        self.indexed_data = []
        self.create_mapping(path, min_sent_lenght, max_sent_lenght)
        self.indices = np.arange(len(self.Ind2word))

    def __len__(self):
        return len(self.indexed_data)

    def create_mapping(self, path, min_sent_lenght, max_sent_lenght):
        word2Ind = {self.pad_token: 0, '<s>': 1, '</s>': 2}
        num_corpus_words = 3
        with open(path) as file:
            for line in file:
                tokenized_line = line.strip().split()
                if len(tokenized_line) < min_sent_lenght or len(tokenized_line) > max_sent_lenght:
                    continue
                for word in tokenized_line:
                    if word not in word2Ind:
                        word2Ind[word] = num_corpus_words
                        num_corpus_words += 1
                tokenized_line = ['<s>'] + tokenized_line + ['</s>']
                indexed_tokenized_line = [word2Ind[word] for word in tokenized_line]
                self.indexed_data.append(indexed_tokenized_line)
        self.Ind2word = dict(enumerate(word2Ind))

    def __getitem__(self, index):
        '''
            __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalize source and
            target values using the vocabulary objects we created in __init__
        '''
        sent_idxs = self.indexed_data[index]
        return sent_idxs