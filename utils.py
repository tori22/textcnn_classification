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

def gen_vocab(path, path2):
    df_all = pd.read_csv(path, encoding='utf-8')
    contents = set(list(df_all['content']))
    vocab_set = set()
    vocab_set.add('none')
    if os.path.exists(path2):
        vocab_dict = pickle.load(open(path2,'rb'))
    else:
        for content in contents:
            for word in contend:
                vocab_set.add(word)
        vocab_dict = {}
        for index, value in enumerate(list(vocab_set)):
            vocab_dict[value] = index
        pickle.dump(vocab_dict,open(path2,'wb'))
    return vocab_dict

def format_padding(data_dict, vocab_dict,nedded_len):
    sentences = pd.Series(list(data_dict.keys()))
    sentences = sentences.apply(lambda sen:sen[:nedded_len])
    sentences = sentences.apply(lambda sen:[vocab_dict[vocab] if vocab in vocab_dict.keys() else vocab_dict['none'] for vocab in sen])
    sentences = sentences.apply(lambda sen:sen+[vocab_dict['none']] * (nedded_len-len(sen)))
    data = list(zip(sentences, data_dict.values()))
    return data

