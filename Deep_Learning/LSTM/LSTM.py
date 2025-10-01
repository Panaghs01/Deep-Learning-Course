import os
import re
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import Counter
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from torcheval.metrics.functional import bleu_score
import cv2
"""Dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k """

#--------------------------------PREPROCESSING--------------------------------#

#building a vocabulary
class vocab:
    #threshold for most common frequency
    def __init__(self, theta = 3):
        self.theta = theta
        self.decoded = {0: "pad", 1: "startofseq", 2: "endofseq", 3: "unk"}
        self.encoded = {v: k for k, v in self.decoded.items()}
        self.index = 4    
        
    def __len__(self):
        return len(self.decoded)
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)
     
        for word, freq in frequencies.items():
            if freq >= self.theta:
                self.encoded[word] = self.index
                self.decoded[self.index] = word
                self.index += 1
    
    def tokenizer(self, text):
        text = text.lower()
        tokens = re.findall(r"\w+", text)
        return tokens



class flickr8k(Dataset):
    def __init__(self,data,vocab,transform = None):
        self.flattened = []
        self.transform = transform
        self.vocab = vocab
        #flatten the dictionary
        for imgID,captions in data.items():
            for caption in captions:
                self.flattened.append((imgID,caption))
                
    #Encode text based on this vocabulary instance
    def numericalize(self, text):
        tokens = self.tokenizer(text)
        numericalized = []
        for token in tokens:
            if token in self.vocab.encoded:
                numericalized.append(self.vocab.encoded[token])
            else:
                numericalized.append(self.vocab.encoded["unk"])
        return numericalized
            
    
    def __len__(self):
        return len(self.flattened)
    
    def __getitem__(self, index):
        img_id, caption = self.flattened[index]
        img_path = os.path.join(IMAGES_DIR, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        numerical_caption = [self.vocab.encoded["startofseq"]]
        numerical_caption += self.numericalize(caption)
        numerical_caption.append(self.vocab.encoded["endofseq"])
        
        return image, torch.tensor(numerical_caption, dtype=torch.long)
        

    def tokenizer(self, text):
        text = text.lower()
        tokens = re.findall(r"\w+", text)
        return tokens

def tokenizetxt(csv_file):
    imgid2captions = {}
    with open(csv_file, "r", encoding="utf-8") as f:
        df = pd.read_csv(f)
        for idx, row in df.iterrows():
            img_id, caption = row['image'], row['caption']
            if img_id not in imgid2captions:
                imgid2captions[img_id] = []
            imgid2captions[img_id].append(caption)
    return imgid2captions

def decodetxt(tensor,vocab):
    return " ".join([vocab.decoded[token.item()] for token in tensor if token.item() in vocab.decoded])

def pad_batch(batch):
    #calculate the largest sequence in the batch and pad 
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]
    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)
 
    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = cap[:end]
 
    images = torch.stack(images, dim=0)
    return images, padded_captions, lengths

#--------------------------------PREPROCESSING--------------------------------#

#----------------------------------Networks-----------------------------------#
class CNN(nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256,5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256*12*12, 2048)
        self.fc2 = nn.Linear(2048, embedding_dim)
        
    def forward(self,x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(F.elu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool(F.elu(self.conv3(x)))
        x = self.bn3(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
class LSTM(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size,num_layers = 1, dropout = 0.3):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                          num_layers = num_layers)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x,caps):
        captions = caps[:,:-1]
        embedded = self.embedding(captions)
        x = x.unsqueeze(1)
        lstm_in = torch.cat((x,embedded),dim=1)
        output, hidden = self.lstm(lstm_in)
        output = self.dropout(output)
        output = self.ln(output)
        output = F.elu(self.fc1(output))
    
        return output

class combination(nn.Module):
    def __init__(self,cnn,lstm):
        super().__init__()
        self.cnn = cnn
        self.lstm = lstm
        
    def forward(self,images,captions):
        cnn_out = self.cnn(images)
        lstm_out = self.lstm(cnn_out,captions)
        
        return lstm_out
    
def train(model,dataloader,epoch):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")
    for images, captions, _lengths in progress_bar:
 
        optimizer.zero_grad()
        outputs = model(images, captions)
        
        outputs = outputs[:, 1:, :].contiguous()
        outputs = outputs.view(-1, vocab_size)
        targets = captions[:, 1:].contiguous()
        targets = targets.view(-1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 0.8)
        
        optimizer.step()
 
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(model,dataloader,vocab):
    model.eval()
    all_pred_caps = []
    all_truth_caps = []
    with torch.no_grad():
        for images,captions,length in dataloader:

            outputs = model(images,captions)
            prediction = torch.argmax(outputs,dim=-1)
            
            pred_caps = [decodetxt(txt, vocab) for txt in prediction]
            truth_caps = [decodetxt(txt, vocab) for txt in captions]
            all_pred_caps.extend(prediction)
            all_truth_caps.extend(truth_caps)
        plt.figure(figsize = (20,10))
        result = bleu_score(all_pred_caps,all_truth_caps)
        print(f"Score: {result}")
        for i in range(5):
            plt.subplot(5,1,i+1)
            image = images[i].permute(1,2,0).cpu().numpy()

            plt.imshow(image)
            plt.axis('off')
            
            plt.title(f"Predicted label: {pred_caps[i]}\nGround truth: {truth_caps[i]}",fontsize = 10)
        plt.show()
#----------------------------------Networks-----------------------------------#

#parameters
LOAD_VOCAB = False
LOAD_MODEL = True
EMBED_DIM = 64
HIDDEN_DIM = 128
LEARNING_RATE = 0.01
BATCH_SIZE = 128
EPOCHS = 5
SEED = 100

IMAGES_DIR = "flickr8k/Images"
TOKENS_FILE = "flickr8k/captions.txt"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


data = tokenizetxt(TOKENS_FILE)
vocab = vocab()
all_caps = []
for captions in data.values():
    all_caps.extend(captions)


vocab.build_vocabulary(all_caps)


vocab_size = len(vocab)

transform = transforms.Compose([
  transforms.Resize((128, 128)),
  transforms.ToTensor()
])

dataset = flickr8k(data, vocab, transform = transform)

trainset , testset = torch.utils.data.random_split(dataset, [0.8,0.2])

train_loader = DataLoader(
  trainset,
  batch_size=BATCH_SIZE,
  shuffle=True,
  collate_fn=pad_batch,
  drop_last=False,
  num_workers=0
)
test_loader = DataLoader(
  testset,
  batch_size=BATCH_SIZE,
  shuffle=False,
  collate_fn=pad_batch,
  drop_last=False,
  num_workers=0
)


cnn = CNN(EMBED_DIM)
lstm = LSTM(EMBED_DIM,HIDDEN_DIM,vocab_size)
model = combination(cnn, lstm)

criterion = nn.CrossEntropyLoss(ignore_index=vocab.encoded["pad"])
optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

if not LOAD_MODEL:
    model.train()
    for epoch in range(EPOCHS):
        train_loss = train(model,train_loader,epoch)
        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}")
    torch.save(model, "lstm.pth")
    validate(model, test_loader,vocab)
else:
    model = torch.load("./lstm.pth",weights_only=False)
    validate(model,test_loader,vocab)

#print(f"average loss of the model {avg_loss}")









