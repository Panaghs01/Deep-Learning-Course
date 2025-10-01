"""
Datasets:
https://www.kaggle.com/datasets/prasunroy/natural-images?select=natural_images
https://www.kaggle.com/datasets/nelyg8002000/commercial-aircraft-dataset


"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from customDataset import aircraft
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time


#model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64,128,5)
        self.fc1 = nn.Linear(128*24*24, 84) #3 conv layers, 128 pixels image, (128-4)/2 -> (62-4)/2 -> (29-4)/2 -> 12
        self.fc2 = nn.Linear(84, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        #MUST NOT USE RELU JUST BEFORE SIGMOID!!!  x<0 -> relu = 0, sigmoid(0) = 0.5. LOSS OF INFORMATION!!
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.sigmoid(x)


#model initialization
net = Net()

#parameters
batch_size = 32
learning_rate = 0.0001
epochs = 2

#Data import
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 128x128
    transforms.ToTensor(),          # Convert PIL image to tensor
])

dataset = aircraft(csv_file = 'labels.csv',root_dir = 'airplanes',transform = transform)

train = round(len(dataset)*0.8)
test = round(len(dataset)*0.2)

train_set,test_set = torch.utils.data.random_split(dataset,[train,test])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

classes = ('not plane','plane')

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train(model,loader):
    start = time.time()
    #Training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader):
    
            # get the inputs, data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.float()
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize , squeeze outputs to fit 
            outputs = net(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    end = time.time()
    print('elapsed time:',end-start)

def overfit(model,loader):
    start = time.time()
    #Training
    criterion = nn.BCEWithLogitsLoss()     #Using BCEWithLogitsLoss instead of simple BCE for the weight arguement
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
    
        running_loss = 0.0
        data = iter(loader)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = next(data)
        labels = labels.float()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize , squeeze outputs to fit 
        outputs = net(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(f'[{epoch + 1}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0

    print('Finished Training')
    end = time.time()
    print('elapsed time:',end-start)
# train(net,trainloader)
# torch.save(net.state_dict(),'my_cnn_final.pth')

#overfit(net,testloader)

net.load_state_dict(torch.load('my_cnn_final.pth'))
# get some random training images
dataiter = iter(testloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print('Ground truth:\n',labels.numpy().reshape(4,8))


outputs = net(images)

predicted = (outputs > 0.5).float()
predicted = predicted.numpy().astype(int)
print('Predictions:\n',predicted.reshape(4,8))


#evaluation
net.eval()

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

#No gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        predictions = (outputs > 0.5).float()
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    

y_true = []
y_pred = []
net.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        labels = labels.float()
        outputs = net(images)
        predicted = (outputs >= 0.5).float()

        y_true.extend(labels.numpy())  # Convert to NumPy for sklearn
        y_pred.extend(predicted.numpy())

# Compute Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)


print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'confusion matrix:\n {confusion_matrix(y_true,y_pred)}')
print(f'classification report:\n {classification_report(y_true,y_pred)}')












