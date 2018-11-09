from utils import loadDataset


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


if __name__=='__main__':
    config = Config()
    print('==>configure initailized')



    train = loadDataset(config.TRAIN_DATA_PATH)
    print(train)
    test = loadDataset(config.TEST_DATA_PATH)
    print(test)
