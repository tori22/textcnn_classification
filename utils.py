import jieba
import re
import pandas as pd

def loadDataset(path):
    df_file = pd.read_csv(path, encoding='utf-8')

    data_dict = {}
    for idx, content in enumerate(df_file['content']):
        if content not in data_dict:
            data_dict[content] = [df_file['subject'][idx], df_file['sentiment_value'][idx]]
        else:
            data_dict[content].extend([df_file['subject'][idx], df_file['sentiment_value'][idx]])

    return data_dict



