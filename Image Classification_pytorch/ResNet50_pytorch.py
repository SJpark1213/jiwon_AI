import torch
import torchvision
from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Parameters
seed = 1
torch.manual_seed(seed)

lr = 0.001
momentum = 0.5
train_batch_size = 30
test_batch_size = 30
epochs = 1

# Device
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data Pre processing
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

base_dir = './Dataset/'
all_train_datasets = []
all_test_datasets = []

for dir in os.listdir(base_dir):
    data_dir = os.path.join(base_dir, dir)
    image_datasets = datasets.ImageFolder(data_dir, transforms)
    class_names = image_datasets.classes
    train_size = int(0.8 * len(image_datasets))
    test_size = len(image_datasets) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(image_datasets, [train_size, test_size])
    all_train_datasets.append(train_dataset)
    all_test_datasets.append(test_dataset)

train_dataset = torch.utils.data.ConcatDataset(all_train_datasets)
test_dataset = torch.utils.data.ConcatDataset(all_test_datasets)

train_loader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)

# # Data Check
# def imshow(img):
#     img = img.numpy()
#     plt.imshow(np.transpose(img, (1,2,0)))
#     plt.show()
#
#
# inputs, targets = next(iter(train_loader))
# imshow(inputs[1])


# Residual Block
class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


# Model: conv1, conv2_x, conv3_x, conv4_x, conv5_x, average pool, 1000d fc, softmax
class ResNet50(nn.Module):
    def __init__(self, block = BottleNeck, num_block = [3,4,6,3], num_classes=3):
        super().__init__()

        self.in_channels=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ResNet50().to(device)

# Train Model
loss_func = nn.CrossEntropyLoss(reduction='sum').cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

result_tensor = [[0,0,0],[0,0,0],[0,0,0]]
columns = ['crackT','resinT','normalT']
rows = ['crackP','resinP','normalP']
data = {columns[0]:[0,0,0],columns[1]:[0,0,0],columns[2]:[0,0,0]}
df = pd.DataFrame(data, index=rows)

for epoch in range(1, epochs + 1):
    # Train Mode
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # backpropagation 계산하기 전에 0으로 기울기 계산
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()  # back propagation으로 gradients 계산
        optimizer.step()  # 계산된 parameter update

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))

    # Test mode
    model.eval()  # batch norm이나 dropout 등을 train mode 변환
    test_loss = 0
    correct = 0
    with torch.no_grad():  # autograd engine, 즉 backpropagatin이나 gradient 계산 등을 꺼서 memory usage를 줄이고 속도를 높임
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # pred와 target과 같은지 확인

            for index in range(len(target)):
                result_tensor[pred[index]][target[index]] += 1
                df[columns[target[index]]][rows[pred[index]]] += 1

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Plot Result
print(df)

# Save Model
checkpoint = {'model': ResNet50(),
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')