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
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from pandas.plotting import parallel_coordinates
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

"""https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"""

torch.manual_seed(23)
np.random.seed(23)
random.seed(23)

class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(   
                nn.Linear(30,12),
                nn.Tanh(),
                nn.Linear(12,8),
                nn.Tanh()
                )
        
    def forward(self,x):
        return self.encoder(x)
        
class decoder(nn.Module): 
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
                nn.Linear(8, 12),
                nn.Tanh(),
                nn.Linear(12,30),
                nn.ELU()
                )
        
    def forward(self,x):
        return self.decoder(x)
    
class Autoencoder(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded

def dimensionality_plot(X, y):
    sns.set(style='whitegrid', palette='muted')
    # Initializing TSNE object with 2 principal components
    tsne = TSNE(n_components=2)
    
    # Fitting the data
    X_trans = tsne.fit_transform(X)
    
    plt.figure(figsize=(12,8))
    
    plt.scatter(X_trans[np.where(y == 0), 0], X_trans[np.where(y==0), 1],
                marker='o', color='green', linewidth=1, alpha=0.8, label='Normal')
    plt.scatter(X_trans[np.where(y == 1), 0], X_trans[np.where(y==1), 1],
                marker='o', color='red', linewidth=1, alpha=0.8, label='Fraud')
    plt.legend(loc = 'best')
    
    plt.show()

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




df = pd.read_csv("data/creditcard.csv")

df['Time'] = df['Time'].apply(lambda t: (t/3600) % 24 )

normal_trans = df[df['Class'] == 0].sample(8000)
fraud_trans = df[df['Class'] == 1]

reduced_set = pd.concat([normal_trans,fraud_trans],ignore_index= True)

y = reduced_set['Class']
X = reduced_set.drop('Class', axis=1)

#dimensionality_plot(X, y)


scaler = RobustScaler().fit_transform(X)

# Scaled data
X_scaled_normal = scaler[y == 0]
X_scaled_fraud = scaler[y == 1]

tester = np.append(X_scaled_normal,X_scaled_fraud,axis = 0)
print(tester.shape)

encoder = encoder()
decoder = decoder()
auto_encoder = Autoencoder(encoder, decoder)

optimizer = optim.Adam(auto_encoder.parameters(), lr = 1e-4)
criterion = nn.MSELoss()

train_loader = torch.utils.data.DataLoader(X_scaled_normal,batch_size = 16, shuffle = True)

for epoch in range(200):
    trainloss = train(auto_encoder,train_loader,epoch)
    print(f"Average loss:{trainloss}")

normal_tran_points = encoder(torch.tensor(X_scaled_normal,dtype = torch.float32))
fraud_tran_points = encoder(torch.tensor(X_scaled_fraud,dtype = torch.float32))

with torch.no_grad():
    predictions = auto_encoder(torch.tensor(tester,dtype = torch.float32))
    predictions = predictions.detach().numpy()
    losses = np.mean(np.square(tester - predictions), axis=1)


# Threshold for defining an anomaly
threshold = losses.mean() + 3 * losses.std()
print(f"Anomaly threshold: {threshold.item()}")

# Detecting anomalies
anomalies = (losses > threshold).astype(int)
anom = pd.DataFrame(anomalies)
print("Classification Report:")
print(classification_report(anomalies, y))
print(f"Accuracy score: {accuracy_score(anomalies,y)}")
# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(anomalies, y))

normal_tran_points = normal_tran_points.detach().numpy()
fraud_tran_points = fraud_tran_points.detach().numpy()

# Making as a one collection
encoded_X = np.append(normal_tran_points, fraud_tran_points, axis=0)
y_normal = np.zeros(normal_tran_points.shape[0])
y_fraud = np.ones(fraud_tran_points.shape[0])
encoded_y = np.append(y_normal, y_fraud, axis=0)

dimensionality_plot(encoded_X, encoded_y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_enc_train, X_enc_test, y_enc_train, y_enc_test = train_test_split(encoded_X, encoded_y, test_size=0.3)

# Instance of SVM
svc_clf = SVC()

svc_clf.fit(X_train, y_train)

svc_predictions = svc_clf.predict(X_test)

print("Classification report \n {0}".format(classification_report(y_test, svc_predictions)))
print("Confusion matrix \n {0}".format(confusion_matrix(y_test,svc_predictions)))

lr_clf = LogisticRegression()

lr_clf.fit(X_enc_train, y_enc_train)

# Predict the Test data
predictions = lr_clf.predict(X_enc_test)

print("Accuracy score is : {:.2f}".format(accuracy_score(y_enc_test, predictions)))

print("Classification report \n {0}".format(classification_report(y_enc_test, predictions)))

print("Confusion matrix \n {0}".format(confusion_matrix(y_enc_test,predictions)))















