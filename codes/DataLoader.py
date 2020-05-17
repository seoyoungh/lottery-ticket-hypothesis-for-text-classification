import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random

SEED = 1234 # random seed for reproductivity

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class IMDB_CNN:
    def __init__(self, batch_size, device, path):

        # binary-class
        self.label_size = 2

        DATA_PATH = path + "/data/imdb/"
        self.TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
        self.LABEL = data.LabelField(dtype = torch.float)

        # train 28000 valid 12000 test 10000
        self.train = data.TabularDataset(path = DATA_PATH + 'train.csv', format = 'csv', fields = [('text', self.TEXT), ('label', self.LABEL)], skip_header = True)
        self.test = data.TabularDataset(path = DATA_PATH + 'test.csv', format = 'csv', fields = [('text', self.TEXT), ('label', self.LABEL)], skip_header = True)
        self.train, self.valid = self.train.split(split_ratio = 0.7, random_state = random.seed(SEED))

        MAX_VOCAB_SIZE = 25_000
        self.TEXT.build_vocab(self.train,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = "glove.6B.100d",
                 unk_init = torch.Tensor.normal_)

        self.LABEL.build_vocab(self.train)

        self.train_iter, self.valid_iter, self.test_iter = data.BucketIterator.splits(
        (self.train, self.valid, self.test),
        sort_key = lambda x: x.text,
        batch_size = batch_size,
        device = device)

        self.vocab_size = len(self.TEXT.vocab)

class IMDB_LSTM:
    def __init__(self, batch_size, device, path):

        # binary-class
        self.label_size = 2

        DATA_PATH = path + "/data/imdb/"
        self.TEXT = data.Field(tokenize='spacy', include_lengths=True)
        self.LABEL = data.LabelField(dtype = torch.float)

        # train 28000 valid 12000 test 10000
        self.train = data.TabularDataset(path = DATA_PATH + 'train.csv', format = 'csv', fields = [('text', self.TEXT), ('label', self.LABEL)], skip_header = True)
        self.test = data.TabularDataset(path = DATA_PATH + 'test.csv', format = 'csv', fields = [('text', self.TEXT), ('label', self.LABEL)], skip_header = True)
        self.train, self.valid = self.train.split(split_ratio = 0.7, random_state = random.seed(SEED))

        MAX_VOCAB_SIZE = 25_000
        self.TEXT.build_vocab(self.train,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = "glove.6B.100d",
                 unk_init = torch.Tensor.normal_)

        self.LABEL.build_vocab(self.train)

        self.train_iter, self.valid_iter, self.test_iter = data.BucketIterator.splits(
        (self.train, self.valid, self.test),
        sort_key = lambda x: x.text,
        batch_size = batch_size,
        device = device)

        self.vocab_size = len(self.TEXT.vocab)

class AGNEWS:
    def __init__(self, batch_size, device, path):

        # multi-class
        self.label_size = 4

        DATA_PATH = path + "/data/ag_news/"
        self.TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
        self.LABEL = data.LabelField()

        # train 84000 valid 36000 test 7600
        self.train = data.TabularDataset(path = DATA_PATH + 'train.csv', format = 'csv', fields = [('text', self.TEXT), ('label', self.LABEL)], skip_header = False)
        self.test = data.TabularDataset(path = DATA_PATH + 'test.csv', format = 'csv', fields = [('text', self.TEXT), ('label', self.LABEL)], skip_header = False)
        self.train, self.valid = self.train.split(split_ratio = 0.7, random_state = random.seed(SEED))

        MAX_VOCAB_SIZE = 25_000
        self.TEXT.build_vocab(self.train,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = "glove.6B.100d",
                 unk_init = torch.Tensor.normal_)

        self.LABEL.build_vocab(self.train)

        self.train_iter, self.valid_iter, self.test_iter = data.BucketIterator.splits(
        (self.train, self.valid, self.test),
        sort_key = lambda x: x.text,
        batch_size = batch_size,
        device = device)

        self.vocab_size = len(self.TEXT.vocab)
