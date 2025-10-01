import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset,DataLoader,Subset
import time
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report
from torch.utils.data import random_split
import numpy as np



"""Dataset: https://www.kaggle.com/datasets/hetulmehta/website-classification """

pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns',20)
# torch.manual_seed(534)
# np.random.seed(534)

class custom(Dataset):
    def __init__(self,data,maxlen):
        self.data = data
        #Turn labels into numbers
        self.classes = data['Category'].unique()
        mapper = {l: idx for idx,l in enumerate(self.classes)}
        data = data.replace(mapper)
        self.labels = data['Category']

        
        #encoding
        encoded = []
        for idx,row in df.iterrows():
            encoded.append([
                vocab.get(word,vocab['<UNK>'])
                for word in row['text']])
        #maxlen computation
        self.maxlen = maxlen
        
        
        #pad all the data so that the tensors are all the same size
        self.padded = []
        
        for txt in encoded:
            if len(txt) < self.maxlen:
                #add PAD token enough times to reach the max length
                self.padded.append(txt + [vocab['<PAD>']] * (self.maxlen - len(txt)))
            else:
                self.padded.append(txt[:self.maxlen])
                     
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = torch.tensor(self.padded[index],dtype = torch.long)
        label = torch.tensor(self.labels[index], dtype = torch.long)
       
        return sample,label

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,num_layers = 1,dropout = 0.5):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                          num_layers = num_layers,
                          bidirectional = False)

        #Multiply with 2 because of bidirectional rnn, double the dimensions
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64,output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        x = output[:, -1, :]
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)

        return x

def train(rnn,loader):
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for epoch in range(epochs):
        rnn.train()
        for i,batch in enumerate(loader):
            data, labels= batch
            outputs = rnn(data)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
        

        print(f"Epoch [{epoch+1}/{epochs}], Step [{i}], Loss: {loss.item():.6f}")
    print(f"Finished after {(time.time() - start)/60} minutes")

    

#load the data and extract the labels
df = pd.read_csv('preped.csv')
df.drop(columns = 'Unnamed: 0',inplace = True)


#load vocabulary made from prep.py
with open('vocab.json','r') as f:
    vocab = json.load(f)

# Hyperparameters
vocab_size = len(vocab)
embedding_dim = 128
hidden_dim = 256
learning_rate = 0.0001
maxlen = 16



#possible future issue: some samples have BIG AMMOUNTS of padding
dataset = custom(df,maxlen)
output_dim = len(dataset.classes)  # Number of classes

batches = 32
train_dataset, test_dataset = random_split(dataset, [0.75, 0.25])

#sm = Subset(dataset,indices = range(1392))
trainset = Subset(train_dataset, range(len(train_dataset)))
trainload = DataLoader(trainset,batch_size = batches,shuffle = True)

testload = DataLoader(test_dataset,batch_size = batches,shuffle = False)

rnn = RNN(vocab_size,embedding_dim,hidden_dim,output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(),lr = learning_rate, weight_decay= 1e-5)
epochs = 30
start = time.time()

#################################################################################load
rnn.load_state_dict(torch.load("./rnnfinal.pth", weights_only=True))
print(rnn)
start = time.time()
# train(rnn,trainload)
# torch.save(rnn.state_dict(), "./rnnfinal.pth")
print(f"TIME IS {time.time() - start} second!")

rnn.eval()
correct = 0
total = 0

all_preds = []
all_labels = []
with torch.no_grad():
    for texts, labels in testload:
        outputs = rnn(texts)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

accuracy_final = 100 * correct / total
print(f'Accuracy: {accuracy_final:.2f}%')

# Print classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds))

# Print confusion matrix
print("Confusion Matrix:")
conf = confusion_matrix(all_labels, all_preds)
print(conf)
    
disp = ConfusionMatrixDisplay(conf)
disp.plot()
plt.show()

# Plot training loss
plt.plot(range(1, epochs+1), train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.savefig("training_loss.png")
plt.show()












