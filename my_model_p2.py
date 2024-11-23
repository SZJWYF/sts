import torch
import torch.nn as nn

class SimilarityModel(nn.Module):
    def __init__(self, bert_model,external_embed_dim=200, dropout_prob=0.1):
        super(SimilarityModel, self).__init__()
        self.bert = bert_model
        self.dense_layer = nn.Linear(768, external_embed_dim)# 768 or 1024
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.output_layer = nn.Linear(external_embed_dim*3, 1)
        self.output_layer2 = nn.Sigmoid()  # 输出为两个类的 logits

    def forward(self, input_ids, attention_mask, sentence1_words_embeddings, sentence2_words_embeddings):
        # 通过BERT模型获取输出
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # 获取CLS token的嵌入
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size x 768)
        # 通过线性层降维到 (batch_size x external_embed_dim)
        embeddings = self.dense_layer(cls_embeddings)
        dense_relu_output = self.relu(embeddings)
        
        # 使用平均池化将 (batch_size x sequence_length x embedding_dim) 的句子嵌入降维到 (batch_size x embedding_dim)
        reduced_sentence1_embeddings = torch.mean(sentence1_words_embeddings, dim=1)
        reduced_sentence2_embeddings = torch.mean(sentence2_words_embeddings, dim=1)


        # 将三个嵌入连接起来，得到 (batch_size x (200 + 200 + 200))
        combined_embeddings = torch.cat((dense_relu_output, reduced_sentence1_embeddings, reduced_sentence2_embeddings), dim=1)
        
        # 添加 Dropout
        combined_embeddings = self.dropout(combined_embeddings)

        logits = self.output_layer(combined_embeddings)  # (batch_size x 2)
        probabilities = self.output_layer2(logits)

        return probabilities
