import json

from torch.utils.data import Dataset, DataLoader
import os
import re
import random
import torch
import pickle
import jieba
import pandas as pd
from scipy import sparse


def save_data_to_pkl(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")

def load_data_from_pkl(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def remove_punctuation(sentence):
    # 定义一个正则表达式模式，用于匹配所有中英文标点符号
    pattern = r'[^\w\s]'
    # 使用 re.sub() 函数将匹配的标点符号替换为空字符串
    cleaned_sentence = re.sub(pattern, '', sentence)
    return cleaned_sentence


class STSBenchmarkReader:
    """
    STS Benchmark reader to prep the data for evaluation.
    """

    def __init__(self, data_path: str = None):
        assert data_path != None and os.path.isfile(data_path)
        self.data_path = data_path
        data_dict = dict(sent1=[], sent2=[], scores=[])

        with open(data_path) as fopen:
            dataset = list(filter(None, fopen.read().split('\n')))

        sent1 = []
        sent2 = []
        scores = []

        for data in dataset:
            data_list = data.split('\t')
            sent1.append(data_list[5])
            sent2.append(data_list[6])
            scores.append(data_list[4])

        data_dict['sent1'] = sent1
        data_dict['sent2'] = sent2
        data_dict['scores'] = scores
        # sanity check
        assert len(data_dict['sent1']) == len(data_dict['sent2'])
        assert len(data_dict['sent1']) == len(data_dict['scores'])

        self.data = data_dict

        self.sentence_pairs = []
        self.scores = []

        for i in range(len(self.data['sent1'])):
            self.sentence_pairs.append((self.data['sent1'][i], self.data['sent2'][i]))
            self.scores.append(float(self.data['scores'][i])/5.0)

    def get_data(self):
        return self.sentence_pairs, self.scores



class ATECBenchmarkReader:
    """
    ATEC Benchmark reader to prep the data for evaluation.
    """
    def __init__(self, data_path: str = None):
        self.data_path = data_path

        with open(data_path) as fopen:
            dataset = list(filter(None, fopen.read().split('\n')))

        self.sentence_pairs = []
        self.scores = []
        for data in dataset:
            split_data = data.split('\t')
            self.sentence_pairs.append((split_data[0], split_data[1]))
            self.scores.append(float(split_data[2]))

    def get_data(self):
        return self.sentence_pairs, self.scores


class BQBenchmarkReader:
    def __init__(self,data_path: str = None):
        self.data_path = data_path
        self.train_data_path = data_path+'/train.csv'
        self.dev_data_path = data_path+'/dev.csv'

        self.train_sentence_pairs = []
        self.train_scores = []
        self.dev_sentence_pairs = []
        self.dev_scores = []

        train_data = pd.read_csv(self.train_data_path, header=None, sep=',')
        # 循环每一行
        for index, row in train_data.iterrows():
            if row[3] == 'bq_corpus':
                self.train_sentence_pairs.append([row[0], row[1]])
                self.train_scores.append(int(row[2]))

        dev_data = pd.read_csv(self.dev_data_path, header=None, sep=',')
        for index, row in dev_data.iterrows():
            if row[3] == 'bq_corpus':
                self.dev_sentence_pairs.append([row[0], row[1]])
                self.dev_scores.append(int(row[2]))

    def get_data(self):
        return self.train_sentence_pairs, self.train_scores, self.dev_sentence_pairs, self.dev_scores

class ChineseSTSBBenchmarkReader:
    def __init__(self,data_path: str = None):
        self.data_path = data_path
        self.train_data_path = data_path+'/train.txt'
        self.test_data_path = data_path + '/test.txt'
        self.dev_data_path = data_path+'/dev.txt'

        self.train_sentence_pairs = []
        self.train_scores = []
        self.test_sentence_pairs = []
        self.test_scores = []
        self.dev_sentence_pairs = []
        self.dev_scores = []

        with open(self.train_data_path, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                line = line.split('\t')
                self.train_sentence_pairs.append((line[0], line[1]))
                self.train_scores.append(float(line[2])/5)

        with open(self.dev_data_path, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                line = line.split('\t')
                self.dev_sentence_pairs.append((line[0], line[1]))
                self.dev_scores.append(float(line[2])/5)

        with open(self.test_data_path, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                line = line.split('\t')
                self.test_sentence_pairs.append((line[0], line[1]))
                self.test_scores.append(float(line[2])/5)

    def get_data(self):
        return self.train_sentence_pairs, self.train_scores, self.test_sentence_pairs,self.test_scores,self.dev_sentence_pairs, self.dev_scores

class OPPOxiaobuBenchmarkReader:
    def __init__(self,data_path: str = None):
        self.data_path = data_path
        self.train_data_path = data_path+'/train.txt'
        # self.test_data_path = data_path + '/test.txt'
        self.dev_data_path = data_path+'/dev.txt'

        self.train_sentence_pairs = []
        self.train_scores = []
        # self.test_sentence_pairs = []
        # self.test_scores = []
        self.dev_sentence_pairs = []
        self.dev_scores = []

        with open(self.train_data_path, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                line = line.split('\t')
                self.train_sentence_pairs.append((line[0], line[1]))
                self.train_scores.append(float(line[2]))

        with open(self.dev_data_path, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                line = line.split('\t')
                self.dev_sentence_pairs.append((line[0], line[1]))
                self.dev_scores.append(float(line[2]))

        # with open(self.test_data_path, 'r') as f:
        #     for line in f:
        #         print(line)
        #         line = line.replace('\n', '')
        #         line = line.split('\t')
        #         self.test_sentence_pairs.append((line[0], line[1]))
        #         self.test_scores.append(float(line[2]))

    def get_data(self):
        # return self.train_sentence_pairs, self.train_scores, self.test_sentence_pairs,self.test_scores,self.dev_sentence_pairs, self.dev_scores
        return self.train_sentence_pairs, self.train_scores, self.dev_sentence_pairs, self.dev_scores


class AFQMCBenchmarkReader:
    def __init__(self,data_path: str = None):
        self.data_path = data_path
        self.train_data_path = data_path+'/train.txt'
        self.dev_data_path = data_path+'/dev.txt'

        self.train_sentence_pairs = []
        self.train_scores = []
        self.dev_sentence_pairs = []
        self.dev_scores = []

        with open(self.train_data_path, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                line = line.split('\t')
                self.train_sentence_pairs.append((line[0], line[1]))
                self.train_scores.append(float(line[2]))

        with open(self.dev_data_path, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                line = line.split('\t')
                self.dev_sentence_pairs.append((line[0], line[1]))
                self.dev_scores.append(float(line[2]))

    def get_data(self):
        return self.train_sentence_pairs, self.train_scores, self.dev_sentence_pairs, self.dev_scores
   
        

class LCQMCBenchmarkReader:
    """
    LCQMC Benchmark reader to prep the data for evaluation.
    """
    def __init__(self, data_path: str = None):
        self.train_data_path = data_path + '/LCQMC_train.json'
        self.dev_data_path = data_path + '/LCQMC_dev.json'
        self.test_data_path = data_path + '/LCQMC_test.json'

        with open(self.train_data_path) as fopen:
            train_lines = fopen.readlines()
        with open(self.dev_data_path) as fopen:
            dev_lines = fopen.readlines()
        with open(self.test_data_path) as fopen:
            test_lines = fopen.readlines()

        self.train_sentence_pairs = []
        self.train_scores = []
        for line in train_lines:
            line_json = json.loads(line)
            self.train_sentence_pairs.append((line_json['sentence1'], line_json['sentence2']))
            self.train_scores.append(float(line_json['gold_label']))
        self.dev_sentence_pairs = []
        self.dev_scores = []
        for line in dev_lines:
            line_json = json.loads(line)
            self.dev_sentence_pairs.append((line_json['sentence1'], line_json['sentence2']))
            self.dev_scores.append(float(line_json['gold_label']))
        self.test_sentence_pairs = []
        self.test_scores = []
        for line in test_lines:
            line_json = json.loads(line)
            self.test_sentence_pairs.append((line_json['sentence1'], line_json['sentence2']))
            self.test_scores.append(float(line_json['gold_label']))

    def get_data(self):
        return self.train_sentence_pairs, self.train_scores, self.dev_sentence_pairs, self.dev_scores, self.test_sentence_pairs, self.test_scores



class STSDataset(Dataset):
    def __init__(self, sentence_pairs, scores, tokenizer, sentences_dict):
        self.sentence_pairs = sentence_pairs
        self.scores = scores
        self.tokenizer = tokenizer
        self.sentences_dict = sentences_dict
        
        if len(sentences_dict) == 0:
            self.sentence_word_flag = False
        else:
            self.sentence_word_flag = True
        

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        sent1, sent2 = self.sentence_pairs[idx]
        sent1 = remove_punctuation(sent1)
        sent2 = remove_punctuation(sent2)

        score = self.scores[idx]
        # 编码句子对
        inputs = self.tokenizer(sent1, sent2, return_tensors='pt', max_length=512, truncation=True,
                                padding='max_length')

        # 提取 input_ids 和 attention_mask
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # 需要mask的tokens数量
        num_to_mask = int(torch.sum(attention_mask) * 0.15)

        # 获取有效词元的索引
        valid_token_indices = torch.where(attention_mask[0] == 1)[0]

        # 排除特殊词元 ([CLS], [SEP], [PAD])
        special_token_ids = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}
        valid_token_indices = [idx for idx in valid_token_indices if input_ids[0][idx].item() not in special_token_ids]

        # 随机选择要mask的索引
        masked_indices = torch.tensor(random.sample(valid_token_indices, num_to_mask))
        masked_indices = masked_indices.long()

        # 创建mask
        input_ids[0][masked_indices] = self.tokenizer.mask_token_id
        if self.sentence_word_flag:
            return inputs, score, self.sentences_dict[sent1].toarray(), self.sentences_dict[sent2].toarray()
        else:
            return inputs, score, [0], [0]

class STSTestDataset(Dataset):
    def __init__(self, sentence_pairs, scores, tokenizer, sentences_dict):
        self.sentence_pairs = sentence_pairs
        self.scores = scores
        self.tokenizer = tokenizer
        self.sentences_dict = sentences_dict
        if len(sentences_dict) == 0:
            self.sentence_word_flag = False
        else:
            self.sentence_word_flag = True

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        sent1, sent2 = self.sentence_pairs[idx]
        sent1 = remove_punctuation(sent1)
        sent2 = remove_punctuation(sent2)

        score = self.scores[idx]
        # 编码句子对
        inputs = self.tokenizer(sent1, sent2, return_tensors='pt', max_length=512, truncation=True,
                                padding='max_length')
        if self.sentence_word_flag:
            return inputs, score, self.sentences_dict[sent1].toarray(), self.sentences_dict[sent2].toarray()
        else:
            return inputs, score, [0], [0]
        

class STSDataset2(Dataset):
    def __init__(self, sentence_pairs, scores, tokenizer):
        self.sentence_pairs = sentence_pairs
        self.scores = scores
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        sent1, sent2 = self.sentence_pairs[idx]
        sent1 = remove_punctuation(sent1)
        sent2 = remove_punctuation(sent2)

        score = self.scores[idx]
        # 编码句子对
        inputs = self.tokenizer(sent1, sent2, return_tensors='pt', max_length=512, truncation=True,
                                padding='max_length')

        # 提取 input_ids 和 attention_mask
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # 需要mask的tokens数量
        num_to_mask = int(torch.sum(attention_mask) * 0.15)

        # 获取有效词元的索引
        valid_token_indices = torch.where(attention_mask[0] == 1)[0]

        # 排除特殊词元 ([CLS], [SEP], [PAD])
        special_token_ids = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}
        valid_token_indices = [idx for idx in valid_token_indices if input_ids[0][idx].item() not in special_token_ids]

        # 随机选择要mask的索引
        masked_indices = torch.tensor(random.sample(valid_token_indices, num_to_mask))

        # 创建mask
        input_ids[0][masked_indices] = self.tokenizer.mask_token_id
        return inputs, score





