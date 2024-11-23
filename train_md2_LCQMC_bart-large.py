
from transformers import BertTokenizer,BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModel


from DataUtil import *
from my_model_p2 import *
import torch
from tqdm import tqdm
from torch import optim
from torch.utils.data import Dataset, DataLoader

def save_data_to_pkl(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")

def load_data_from_pkl(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lcqmc_benchmarkReader= LCQMCBenchmarkReader(data_path="LCQMC")
train_sentence_pairs, train_scores, dev_sentence_pairs, dev_scores, test_sentence_pairs, test_scores = lcqmc_benchmarkReader.get_data()


bert_file_path = "bart-large-chinese"
print('will load bert is '+ bert_file_path)


tokenizer = AutoTokenizer.from_pretrained(bert_file_path)
bert_model = AutoModel.from_pretrained(bert_file_path).to(device)

sentences_dict = load_data_from_pkl('LCQMC/LCQMC_sentence_dict.pkl')

lcqmc_train_Dataset = STSDataset(train_sentence_pairs, train_scores, tokenizer, sentences_dict)
lcqmc_test_Dataset = STSTestDataset(test_sentence_pairs, test_scores, tokenizer, sentences_dict)
lcqmc_dev_Dataset = STSTestDataset(dev_sentence_pairs, dev_scores, tokenizer, sentences_dict)

lcqmc_train_loader = DataLoader(lcqmc_train_Dataset, batch_size=16, shuffle=True)
lcqmc_test_loader = DataLoader(lcqmc_test_Dataset, batch_size=16, shuffle=False)
lcqmc_dev_loader = DataLoader(lcqmc_dev_Dataset, batch_size=16, shuffle=False)


print("Model loading...")
myModel = SimilarityModel(bert_model).to(device)# 确保模型在GPU上
# myModel.load_state_dict(torch.load('/data/szj/sts/transformers/SZJ_Model/STSModel.pth'))
criterion = nn.BCELoss()

def train_model(model, train_loader, test_loader,dev_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        # 使用 tqdm 包裹数据加载器，显示进度条
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            inputs, scores, sentence1_words_embeddings, sentence2_words_embeddings = batch
            input_ids = inputs['input_ids'].squeeze(1).to(device)  # 移动到GPU
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            scores = scores.float()
            scores = scores.to(device)  # 移动到GPU
            scores = scores.unsqueeze(1)
            sentence1_words_embeddings = sentence1_words_embeddings.float()
            sentence2_words_embeddings = sentence2_words_embeddings.float()
            sentence1_words_embeddings = sentence1_words_embeddings.to(device)
            sentence2_words_embeddings = sentence2_words_embeddings.to(device)

            outputs = model(input_ids, attention_mask,
                            sentence1_words_embeddings,
                            sentence2_words_embeddings)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

        evaluate_model_p(myModel, test_loader)
        evaluate_model_p(myModel, dev_loader)
        print("accuracy_list:"+str(accuracy_list))

# Evaluation function
def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, scores, sentence1_words_embeddings, sentence2_words_embeddings = batch
            input_ids = inputs['input_ids'].squeeze(1).to(device)  # 移动到GPU
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            scores = scores.to(device)  # 移动到GPU
            sentence1_words_embeddings = sentence1_words_embeddings.to(device)
            sentence2_words_embeddings = sentence2_words_embeddings.to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            sentence1_words_embeddings=sentence1_words_embeddings,
                            sentence2_words_embeddings=sentence2_words_embeddings)
            loss = criterion(outputs, scores)
            total_loss += loss.item()
    test_loss = total_loss / len(data_loader)
    print(f"Test Loss: {test_loss}")
    return test_loss

def evaluate_model_p(model, data_loader):
    model.eval()
    ture_num = 0
    total_num = 0
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, scores, sentence1_words_embeddings, sentence2_words_embeddings = batch
            input_ids = inputs['input_ids'].squeeze(1).to(device)  # 移动到GPU
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            scores = scores.float()
            scores = scores.to(device)  # 移动到GPU
            scores = scores.unsqueeze(1)
            sentence1_words_embeddings = sentence1_words_embeddings.float()
            sentence2_words_embeddings = sentence2_words_embeddings.float()
            sentence1_words_embeddings = sentence1_words_embeddings.to(device)
            sentence2_words_embeddings = sentence2_words_embeddings.to(device)

            outputs = model(input_ids, attention_mask,
                            sentence1_words_embeddings,
                            sentence2_words_embeddings)
            loss = criterion(outputs, scores)
            total_loss += loss.item()
            for i in range(len(outputs)):
                if float(outputs[i]) > 0.5:
                    score = 1
                else:
                    score = 0
                if int(scores[i]) == score:
                    ture_num += 1
                total_num += 1
    test_loss = total_loss / len(data_loader)
    print(f"Test Loss: {test_loss}")
    print('Accuracy: '+str(ture_num/total_num))
    accuracy = ture_num/total_num
    loss_list.append(test_loss)
    accuracy_list.append(accuracy)
    global best_loss
    global best_model

    if test_loss <= best_loss:
        best_loss = test_loss
        best_model = model
        torch.save(myModel, 'bart-large-LCQMC.pth')
    print("best:"+str(best_loss))
    

# Training the model
print("Training...")
best_loss=1000.0
loss_list = []
accuracy_list = []
print("Training...")
optimizer = optim.Adam(myModel.parameters(), lr=1e-5)
train_model(myModel, lcqmc_train_loader, lcqmc_test_loader,lcqmc_dev_loader,criterion, optimizer, epochs=15)
myModel = best_model
print('-------------')
print("loss_list:"+str(loss_list))
print("accuracy_list:"+str(accuracy_list))
