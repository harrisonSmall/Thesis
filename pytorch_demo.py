import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt

# # Tensor operations example
# # x = torch.empty(2, 3, 3) # empty 3x3 tensor, can specify any dimensions
# # print(x)
# x = torch.rand(2, 2)
# y = torch.rand(2, 2, dtype=torch.float16) # random 2x2, can do .zeros or .ones, or specify datatype
# print(y.dtype) #float32 by default
# print(y.size())
# print(x + y) # or torch.add(x, y) or y.add_(x) for inplace (trailing _ means inplace)
# z = torch.tensor([2.5, 0.1]) # specified tensor
# # can use slicing like numpy array: x[:,1]
# # .item() will return the contents if only 1 value
# # reshape
# print(y.view(4)) # .view(-1, n) will automatically size the other dimension
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b) # If the tensor on the cpu not gpu they will share the same location so careful with modifications
# a.add_(1)
# print(b)
# c = torch.from_numpy(b)
# print(c)

if torch.cuda.is_available():
    print('yes')
    # Moves operations to gpu, shouldn't matter until using goliath?

# lower level back propagation with single value tensors
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)
#forward pass
y_hat = w * x
loss = (y_hat - y)**2
print(loss)
#backward pass
loss.backward()
print('gradient of weights:', w.grad)
# Update weights and repeat
with torch.no_grad():
    w -= w.grad # * learning rate
w.grad.zero_()

#------------------CNN DEMO----------------------------

# Hyperparameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# Transform images to Tensors normalised to [-1, 1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Get data
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Data loaders (easy way to parse to network)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Network definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Initialise layers
        self.full1 = nn.Linear(784, 50)
        self.full2 = nn.Linear(50, 10)

    def forward(self, x):
        #Activation functions
        x = x.view(-1, 784)
        x = F.relu(self.full1(x))
        x = F.relu(self.full2(x))
        return F.log_softmax(x)

model = Net()

criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Do device stuff?

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backwards pass
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}, Loss: {loss.item():.4f} %')

print('Training Finished')

# Testing loop
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        outputs = model(images)
        #max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')