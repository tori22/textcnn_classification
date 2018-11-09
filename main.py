from utils import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model import *


class Config(object):
    def __init__(self):
        self.TRAIN_DATA_PATH = './data/train_data.csv'
        self.DATA_ALL = './data/train.csv'
        self.TEST_DATA_PATH = './data/test_data.csv'
        self.SUB_TO_LABEL_PATH = './source/sub_to_label.pkl'
        self.WORD_FREQUENCY_DICT = './source/word_frequency_dict.pkl'
        self.VOCAB_DICT_PATH = './source/vocab_dict.pkl'
        self.SENTENCE_LEN = 80
        self.CUT = 1
        self.BATCH_SIZE = 300
        self.EMBEDDING_DIM = 512
        self.HIDDING_DIM = 200
        self.HIDDING_DIM_2 = 10
        self.KERNEL_SIZE = [2,3,4,5]
        self.KERNEL_SIZE_2 = 3
        self.FC_HID_DIM = 128
        self.LABEL_NUM_1 = 10
        self.LABEL_NUM_2 = 3
        self.USE_GPU = True
        self.Epoch = 30
        self.TEST_STEP = 10
        self.LEARNING_RATE = 0.001
        self.CUT_VOCAB = False


class dataset(Dataset):
    def __init__(self, train_data, sub_to_label):
        self.train_data = train_data
        self.sub_to_label = sub_to_label
    def __getitem__(self, index):
        line = self.train_data[index]
        sentence = line[0]
        sub_senti = line[1]

        label_1 = [np.float32(0)]*config.LABEL_NUM_1
        label_2 = [np.float32(0)]*config.LABEL_NUM_2
        for i in range(0,len(sub_senti),2):
            sub = sub_senti[i]
            sentiment_value = sub_senti[i+1]
            label_1[self.sub_to_label[sub]] = np.float32(sentiment_value)
            label_2[self.sub_to_label[sub]] = np.float32(1)
        return np.array(sentence), np.array(label_1), np.array(label_2)
    def __len_(self):
        return len(self.train_data)

if __name__=='__main__':
    config = Config()
    print('==>configure initailized')

    train_data_dict = loadDataset(config.TRAIN_DATA_PATH)
    test_data_dict = loadDataset(config.TEST_DATA_PATH)
    sub_to_label,label_to_sub = gen_subject_dict(config.DATA_ALL, config.SUB_TO_LABEL_PATH)

    vocab_dict = gen_vocab(config.DATA_ALL, config.VOCAB_DICT_PATH)
    train_data = format_padding(train_data_dict, vocab_dict,config.SENTENCE_LEN)
    test_data = format_padding(test_data_dict, vocab_dict, config.SENTENCE_LEN)
    train_dataset = dataset(train_data, sub_to_label)
    train_dataloader = DataLoader(train_dataset, batch_size = config.BATCH_SIZE, drop_last = True, shuffle = True)
    test_dataset = dataset(test_data, sub_to_label)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_data), drop_last = False, shuffle = False)
    config.VOCAB_SIZE = len(vocab_dict)


    net = TextCnn(config)
    if config.USE_GPU:
        net.cuda()
    print('==> model initailized')


