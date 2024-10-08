#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/6/27 17:08
# @Author : 桐
# @QQ:1041264242
# 注意事项：
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 判断是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 80  # 80轮
batch_size = 32  # 32步长
learning_rate = 0.01  # 学习率0.01

# # 图像预处理
# transform = transforms.Compose([
#     transforms.Pad(4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32),
#     transforms.ToTensor()])
# # 图像预处理
# transform = transforms.Compose([
#     transforms.Resize(64),  # 将图像大小变换为64x64
#     transforms.ToTensor()  # 将PIL图像或NumPy ndarray转换为torch.Tensor
# ])
#
# # CIFAR-10 数据集下载
# train_dataset = torchvision.datasets.CIFAR10(root='../data/',
#                                              train=True,
#                                              transform=transform,
#                                              download=True)
#
# test_dataset = torchvision.datasets.CIFAR10(root='../data/',
#                                             train=False,
#                                             transform=transforms.ToTensor())
#
# # 数据载入
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)

# 图像预处理（这里保留了之前的transform，您可以根据需要调整）

transform = transforms.Compose([
    transforms.Resize(64),  # 调整图像大小为64x64
    transforms.ToTensor()  # 将图像转换为torch.Tensor
])

# 从本地文件夹加载训练和测试数据
train_dataset = torchvision.datasets.ImageFolder(root=r'/remote-home/cs_acmis_sdtw/PICS/PIcs_Resnet/TrainData2', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=r'/remote-home/cs_acmis_sdtw/PICS/PIcs_Resnet/TestData2', transform=transform)

# 数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# for i, (images, labels) in enumerate(train_loader):
#     images = images
#     labels = labels
#     print(images[0].shape)
#     print(labels)
#     break

# 3x3 卷积定义
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Resnet 的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet定义
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = ResNet(ResidualBlock, [2, 2, 2]).to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.002)


# 更新学习率
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 训练数据集
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # print(images[0].shape)

        print(images.shape)
        # Forward pass
        outputs = model(images)
        print(labels)
        print(outputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i + 1) % 100 == 0:
    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # 延迟学习率
    if (epoch + 1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# 测试网络模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# S将模型保存
torch.save(model.state_dict(), 'resnet_time_test.ckpt')

