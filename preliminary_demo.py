import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import glob
import os
from os import walk
import matplotlib

# Variables
data_dir = 'C:/Users/Harrison/Desktop/Thesis/Resources/rfracture-master/rfracture-master/First_100'
learning_rate = 0.001
epochs = 5

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load Y data (output permeability tensors)
df = pd.read_csv('C:/Users/Harrison/Desktop/Thesis/Resources/rfracture-master/rfracture-master/MC_res.csv')
df = df[:200] # take only first 200 points
df = df.loc[df['Re'] == 0.1] # remove Re = 0.1 points
df = df[['P11', 'P12', 'P21', 'P22']] # take only permeability tensor
y_train = df[:80]
y_test = df[80:100]
print(y_test.shape)

# dataset = datasets.DatasetFolder('./data')

# Load X data (input arrays/ fracture planes)
frac_2 = np.load('C:/Users/Harrison/Desktop/Thesis/Resources/rfracture-master/rfracture-master/First_100/frac_2.npy')
print(frac_2.shape)


class NpDataSet(torch.utils.data.Dataset):
    def __init__(self, root_path, tforms):
        self.array_list = [x for x in glob.glob(os.path.join(root_path, '*.npy'))]
        self.transforms = tforms
        self.data_list = []
        for ind in range(len(self.array_list)):
            data_slice_file_name = self.array_list[ind]
            data_i = np.load(data_slice_file_name)
            self.data_list.append(data_i)

    def __getitem__(self, index):
        self.data = np.asarray(self.data_list[index])
        self.data = np.stack((self.data, self.data))
        if self.transforms:
            self.data = self.transforms(self.data)
        return self.data

    def __len__(self):
        return len(self.array_list)


data_train = NpDataSet(data_dir, tforms=None)
x_train_loader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True)


class LeNet(nn.Module):
    # Num_channels=1 for greyscale, 3 rgb. classes = unique class labels.
    def __init__(self, num_channels):
        # Parent constructor
        super(LeNet, self).__init__()

        # Initialise Layers
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=(5, 5))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=200, out_features=500)

        # Initialise activation functions
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        # Initialise softmax classifier
        self.fc2 = nn.Linear(in_features=500, out_features=4)

    def forward(self, x):
        # Pass input through first layer
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # 2nd layer
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # fc layer 1
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        # Softmax
        output = self.fc2(x)

        return output


# Create and train model
model = LeNet(2)
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
steps = len(x_train_loader)
for epoch in range(epochs):
    for i, (arrays, output_tensors) in enumerate(zip(x_train_loader, y_train)):
        arrays = arrays.to(device)
        # output_tensors = output_tensors.to(device)

        #Forward pass
        outputs = model(arrays)
        loss = criterion(outputs, output_tensors)

        #Backward pass
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, epochs, i+1, steps, loss.item()))
