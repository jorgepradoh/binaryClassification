#%% packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from PIL import Image

os.getcwd()

#%%
DatasetFolder = 'C:/Users/jorge/Desktop/github/img_processing/dog_muffin'
TrainData = DatasetFolder + '/train'
TestData = DatasetFolder + '/test'

#%% transform, load data

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
    ])

batch_size = 4

trainset = torchvision.datasets.ImageFolder(root=TrainData, transform = transform)
testset = torchvision.datasets.ImageFolder(root=TestData, transform = transform)

trainloader = DataLoader(trainset, batch_size= batch_size, shuffle = True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle = True)
# %% visualize images

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images, nrow=2))

#%%
def convOut(W = None, K = None, P = None, S = None):
    if W is None:  # W := image input volume 
        W = 0
    if K is None:  # K := kernel size
        K = 0
    if P is None:  # P := padding 
        P = 0
    if S is None:  # S := stride
        S = 1

    return ((W-K+2*P)/S) +1 









# %% Neural Network setup
class ImageClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) #out: BS 6, 222, 222
        self.pool = nn.MaxPool2d(2,2) #out: BS 6, 114, 114
        self.conv2 = nn.Conv2d(6, 16, 3) #out: BS 16, 112, 112 
        self.fc1 = nn.Linear(16 * 62 * 62, 256) #after next pool: BS 16, 59, 59
        self.fc11 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU
        
    
    def forward(self, x):
        x = self.conv1(x) #out: BS 6, 254, 254
        x = F.relu(x)
        x = self.pool(x) #out: BS 6, 127, 127
        x = self.conv2(x) #out: BS 16, 125, 125
        x = F.relu(x)
        x = self.pool(x) #out: BS 16, 62.5, 62.5

        x = torch.flatten(x, 1) #out: BS, 16*64*64
        x = self.fc1(x) #out: BS, 128
        x = F.relu(x)
        x = self.fc11(x)
        x = F.relu(x)
        x = self.fc2(x) #out: BS, 64
        x = F.relu(x)
        x = self.fc3(x) #out: BS, 1

        x = self.sigmoid(x)

        return x

#%% init model
model = ImageClassificationNet()      

lr = 0.001

loss_fn = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# %% training
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # zero gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

        # calc losses
        loss = loss_fn(outputs, labels.reshape(-1, 1).float())

        # backward pass
        loss.backward()
        
        # update weights
        optimizer.step()
        
    if i % 100 == 0:
        print(f'Epoch {epoch}/{NUM_EPOCHS}, Step {i+1}/{len(trainloader)},'
                f'Loss: {loss.item():.4f}')
# %% test
y_test = []
y_test_hat = []
for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    
    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, y_test_hat)
print(f'Accuracy: {acc*100:.2f} %')
