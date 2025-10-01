import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import umap
from pandas.plotting import parallel_coordinates
import random
""" Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"""
   
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(   
               nn.Linear(30,100),
               nn.Tanh(),
               nn.Linear(100, 50),
               nn.Tanh(),
               nn.Linear(50,25),
               nn.Tanh(),
               nn.Linear(25,12),
               nn.Tanh(),
               nn.Linear(12, 8),
               nn.ELU()
               )
        self.decoder = nn.Sequential(
                nn.Linear(8, 12),
                nn.Tanh(),
                nn.Linear(12,25),
                nn.Tanh(),  
                nn.Linear(25,50),
                nn.Tanh(),
                nn.Linear(50, 100),
                nn.Tanh(),
                nn.Linear(100,30),
                nn.ELU())
        
    def forward(self,x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
    
def train(model,dataloader,epoch):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch")
    for item in progress_bar:
        item = item.float()
        recon = model(item)
        loss = criterion(recon, item)
        optimizer.zero_grad()
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)        

        optimizer.step()
 
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def plot(x,y,real):

    dim = TSNE(n_components=2)

    data = dim.fit_transform(x.to_numpy())
    
    plt.figure(figsize=(12,8))

    plt.scatter(data[np.where(y == 0), 0], data[np.where(y==0), 1], marker='o', color='blue', linewidth=1, alpha=0.8, label='Normal')
    plt.scatter(data[np.where(y == 1), 0], data[np.where(y==1), 1], marker='o', color='red', linewidth=1, alpha=0.8, label='Predicted Fraud')
    
    plt.legend(loc = 'best')
    
    plt.show()
    
#PARAMETERS
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 20

#Load the data
df = pd.read_csv("data/creditcard.csv")

df['Time'] = df['Time'].apply(lambda t: (t/3600) % 24 )

cols = list(df.columns)
cols.remove("Class")
for col in cols:
    df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)

x_train = df[df["Class"] == 0].sample(4000)
labels = df["Class"]
y_test = df
y_test = y_test.drop(["Class"],axis = 1)

x_train = x_train.drop(['Class'],axis = 1)
#x_train = x_train.sample(8000)
X_train = x_train.to_numpy()



train_loader = torch.utils.data.DataLoader(X_train,batch_size = BATCH_SIZE,shuffle = True)

model = Autoencoder()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = 0)
for epoch in range(EPOCHS):
    trainloss = train(model,train_loader,epoch)
    scheduler.step(trainloss)
    print(f"Average loss:{trainloss}")
torch.save(model.state_dict(),'auto.pth')


y_test = y_test.to_numpy()
# Anomaly detection
with torch.no_grad():
    predictions = model(torch.tensor(y_test,dtype = torch.float32))
    predictions = predictions.numpy()
    losses = np.mean(np.square(y_test - predictions), axis=1)


# Threshold for defining an anomaly
threshold = losses.mean() + 3 * losses.std()
print(f"Anomaly threshold: {threshold.item()}")

# Detecting anomalies
anomalies = (losses > threshold).astype(int)
anom = pd.DataFrame(anomalies)
print("Classification Report:")
print(classification_report(anomalies, labels))
print(f"Accuracy score: {accuracy_score(anomalies,labels)}")
# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(anomalies, labels))


