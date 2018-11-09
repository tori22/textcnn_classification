import jieba
import re
import pandas as pd
import os
import pickle

def loadDataset(path):
    df_file = pd.read_csv(path, encoding='utf-8')

    data_dict = {}
    for idx, content in enumerate(df_file['content']):
        if content not in data_dict:
            data_dict[content] = [df_file['subject'][idx], df_file['sentiment_value'][idx]]
        else:
            data_dict[content].extend([df_file['subject'][idx], df_file['sentiment_value'][idx]])

    return data_dict

def gen_subject_dict(path, path2):
    df_all = pd.read_csv(path, encoding='utf-8')
    subject = set(list(df_all['subject']))

    sub_to_label = {}
    if os.path.exists(path2):
        sub_to_label = pickle.load(open(path2,'rb'))
    else:
        for sub in subject:
            sub_to_label[sub] = idx
            idx += 1
        pickle.dump(sub_to_label,open(path2,'wb'))

    label_to_sub = dict(zip(sub_to_label.values(), sub_to_label.keys()))
    return sub_to_label, label_to_sub
