#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:32:58 2023

@author: wangsiyu
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

def process_data(data_path,ans_path):
    data=pd.read_csv(data_path,header=0,names=['ID','sentence0','sentence1'])
    ans=pd.read_csv(ans_path,header=None,names=['ID','answer'])['answer']
    df=pd.concat([data,ans],axis=1)
    
    return df


def classification_data(mode):
    if mode == 'train':
        df = process_data('../data/Training_Data/subtaskA_data_all.csv',
                         '../data/Training_Data/subtaskA_answers_all.csv')
    elif mode == 'dev':
        df = process_data('../data/Dev_Data/subtaskA_dev_data.csv',
                         '../data/Dev_Data/subtaskA_gold_answers.csv')
    elif mode == 'test':
        df = process_data('../data/Test_Data/subtaskA_test_data.csv',
                         '../data/Test_Data/subtaskA_gold_answers.csv')
    
    return df

def encode_data(df, tokenizer, question):
    sentences = sum([df.iloc[i, 1:3].tolist() for i in range(len(df))], start=[])
    input_ids = []
    attention_mask = []
    token_type_ids = []

    for sent in sentences:
        encoding = tokenizer.encode_plus(question,sent, max_length=50, truncation=True,
                                         padding="max_length", add_special_tokens=True,
                                         return_attention_mask=True, return_tensors='pt')

        input_ids.append(encoding['input_ids'])
        attention_mask.append(encoding['attention_mask'])
        token_type_ids.append(encoding['token_type_ids'])

    input_ids = torch.cat(input_ids, dim=0).view(len(df), 2, -1)
    attention_mask = torch.cat(attention_mask, dim=0).view(len(df), 2, -1)
    token_type_ids = torch.cat(token_type_ids, dim=0).view(len(df), 2, -1)

    res = dict()
    res['input_ids'] = input_ids
    res['attention_mask'] = attention_mask
    res['token_type_ids'] = token_type_ids

    return res


class ComVEDataset(Dataset):
    def __init__(self,encodings,labels):
        self.encodings=encodings
        self.labels=labels
    
    def __getitem__(self,idx):
        item={key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels']=torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)    