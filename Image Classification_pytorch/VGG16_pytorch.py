import torch
import torchvision
from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def run():
    torch.multiprocessing.freeze_support()

    seed = 1

    lr = 0.001
    momentum = 0.5

    batch_size = 25
    test_batch_size = 25

    epochs = 5

    no_cuda = False
    log_interval = 100

    torch.manual_seed(seed)

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform), batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform),
                         batch_size=test_batch_size, shuffle=False, **kwargs)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Sequential(
                #3 224 128
                nn.Conv2d(3, 64, 3, padding=1),nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 3, padding=1),nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2),
                #64 112 64
                nn.Conv2d(64, 128, 3, padding=1),nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, 3, padding=1),nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2),
                #128 56 32
                nn.Conv2d(128, 256, 3, padding=1),nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, 3, padding=1),nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2),
                #256 28 16
                nn.Conv2d(256, 512, 3, padding=1),nn.LeakyReLU(0.2),
                nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
                nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2),
                #512 14 8
                nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
                nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
                nn.Conv2d(512, 512, 3, padding=1),nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            )
            #512 7

            self.fc1 = nn.Linear(512*7*7,4096)
            self.fc2 = nn.Linear(4096,4096)
            self.fc3 = nn.Linear(4096,10)
            self.softmax = nn.Softmax(dim=1)


        def forward(self, x):
            features = self.conv(x)
            x = features.view(features.size(0), -1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            x = self.softmax(x)
            return x

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(1, epochs + 1):
        print(device)
        # Train Mode
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # backpropagation 계산하기 전에 0으로 기울기 계산
            output = model(data)
            loss = criterion(output, target)
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
                test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()  # pred와 target과 같은지 확인

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


if __name__ =='__main__':
    run()