import torch
import torch.nn as nn

class SimilarityModel(nn.Module):
    def __init__(self, bert_model, external_embed_dim, dropout_prob=0.1):
        super(SimilarityModel, self).__init__()
        self.bert = bert_model
        self.dense_layer = nn.Linear(768, 200)
        self.activation = nn.GELU()  # 使用 GELU 激活函数
        self.layer_norm_bert = nn.LayerNorm(200)
        self.layer_norm_external = nn.LayerNorm(200)
        self.dropout = nn.Dropout(dropout_prob)
        self.external_dense = nn.Linear(external_embed_dim, 200)
        self.fc1 = nn.Linear(600, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.output_layer = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, sentence1_embeddings, sentence2_embeddings):
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size x 768)

        # 处理BERT的CLS嵌入
        embeddings = self.dense_layer(cls_embeddings)
        embeddings = self.activation(embeddings)
        embeddings = self.layer_norm_bert(embeddings)

        # 处理外部嵌入
        sentence1_embeddings = sentence1_embeddings.float()
        # 平均池化
        sentence1_embeddings = torch.mean(sentence1_embeddings, dim=1)
        sentence1_embeddings = self.external_dense(sentence1_embeddings)
        sentence1_embeddings = self.activation(sentence1_embeddings)
        sentence1_embeddings = self.layer_norm_external(sentence1_embeddings)

        sentence2_embeddings = sentence2_embeddings.float()
        sentence2_embeddings = torch.mean(sentence2_embeddings, dim=1)
        sentence2_embeddings = self.external_dense(sentence2_embeddings)
        sentence2_embeddings = self.activation(sentence2_embeddings)
        sentence2_embeddings = self.layer_norm_external(sentence2_embeddings)

        # 融合嵌入
        combined_embeddings = torch.cat((embeddings, sentence1_embeddings, sentence2_embeddings), dim=1)
        combined_embeddings = self.dropout(combined_embeddings)

        combined_embeddings = torch.relu(self.fc1(combined_embeddings))
        combined_embeddings = torch.relu(self.fc2(combined_embeddings))
        # 最后一层使用Sigmoid激活函数
        probabilities = self.output_layer(self.fc3(combined_embeddings))

        return probabilities