from DataUtil import *
LCQMCReader = AFQMCBenchmarkReader('AFQMC')
train_sentence_pairs, train_scores, dev_sentence_pairs, dev_scores = LCQMCReader.get_data()

# sentence_list = LCQMC_train_sentence_pairs+LCQMC_dev_sentence_pairs+LCQMC_test_sentence_pairs
sentence_list = train_sentence_pairs  + dev_sentence_pairs
sentence_list = sentence_list[30000:]
#38650


import pickle
def save_data_to_pkl(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")

def load_data_from_pkl(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

from tqdm import tqdm
from BCEmbedding import EmbeddingModel
import numpy as np
from scipy.sparse import csr_matrix
import re
embedding_model = EmbeddingModel(model_name_or_path="/data/coding/Short-Text-Similarity/bce-embedding")
word_vector_length = 768
# 保证 n 为 50
n = 50
sentence_dict={}
i = 0
for sentence1,sentence2 in tqdm(sentence_list):
    print(str(i) + '/' +str(len(sentence_list)))
    i = i+1
    seg_list1 = list(jieba.cut(remove_punctuation(sentence1), cut_all=True))
    # 获取模型中每个词的向量
    sentence_embedding1 = []
    embeddings =[]
    for word in seg_list1:
        try:
            embeddings.append(embedding_model.encode([word])[0])
        except:
            embeddings.append(np.zeros(word_vector_length))
    if len(sentence_embedding1) > n:
        # 如果超过 50 个词，只取前 50 个词
        sentence_embedding1 = sentence_embedding1[:n]
    else:
        # 如果少于 50 个词，用零向量补充
        padding_length = n - len(sentence_embedding1)
        sentence_embedding1.extend([np.zeros(word_vector_length)] * padding_length)
    # 最终的句子向量矩阵为 50 x 200
    sentence_matrix1 = np.array(sentence_embedding1)
    sentence_matrix1 = csr_matrix(sentence_matrix1)
    sentence_dict[remove_punctuation(sentence1)] = sentence_matrix1

    seg_list2 = list(jieba.cut(remove_punctuation(sentence2), cut_all=True))
    # 获取模型中每个词的向量
    sentence_embedding2 = []
    embeddings =[]
    for word in seg_list2:
        try:
            embeddings.append(embedding_model.encode([word])[0])
        except:
            embeddings.append(np.zeros(word_vector_length))
    if len(sentence_embedding2) > n:
        # 如果超过 50 个词，只取前 50 个词
        sentence_embedding2 = sentence_embedding2[:n]
    else:
        # 如果少于 50 个词，用零向量补充
        padding_length = n - len(sentence_embedding2)
        sentence_embedding2.extend([np.zeros(word_vector_length)] * padding_length)
    # 最终的句子向量矩阵为 50 x 200
    sentence_matrix2 = np.array(sentence_embedding2)
    sentence_matrix2 = csr_matrix(sentence_matrix2)
    sentence_dict[remove_punctuation(sentence2)] = sentence_matrix2

    save_data_to_pkl(sentence_dict, 'AFQMC_sentence_bce_dict_4.pkl')