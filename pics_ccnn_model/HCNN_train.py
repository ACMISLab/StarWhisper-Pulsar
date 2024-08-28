#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/7/2 15:26
# @Author : 桐
# @QQ:1041264242
# 注意事项：
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# 判断是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 100  # 80轮
batch_size = 64  # 32步长
learning_rate = 1e-3  # 学习率0.01


class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.data = []
        self.labels = []

        # 读取文件并存储路径和标签
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                path, label = line.strip().split()
                self.data.append(path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class FeatureDataset(Dataset):  
    def __init__(self, file_path):  
        self.file_path = file_path  
        self.features = []  
          
        # 读取特征文件并存储特征  
        with open(file_path, 'r') as file:  
            for line in file:
                # feature = line.strip().split()  # 假设特征是空格分隔的数值  
                # self.features.append([float(f) for f in feature])
                values = line.strip().split() 
                # 忽略第一个值，只取剩下的部分  
                if len(values) > 1:  
                    feature = [[float(f) for f in values[1:]]]  # 从第二个值开始转换  
                    self.features.append(feature) 
      
    def __len__(self):  
        return len(self.features)  
      
    def __getitem__(self, idx):  
        feature = self.features[idx]  
        # 将特征转换为Tensor  
        feature_tensor = torch.tensor(feature, dtype=torch.float)  
        return feature_tensor  

transform = transforms.Compose([
    transforms.Resize(64),  # 调整图像大小为64x64
    transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图像
    transforms.ToTensor()  # 将图像转换为torch.Tensor
])

# 创建数据集
dataset_train_sub = CustomDataset('/remote-home/cs_acmis_sdtw/PICS/PIcs_Resnet/TrainFile/train_sub.txt', transform=transform)
dataset_train_time = CustomDataset('/remote-home/cs_acmis_sdtw/PICS/PIcs_Resnet/TrainFile/train_time.txt', transform=transform)
dataset_train_profile = FeatureDataset('/remote-home/cs_acmis_sdtw/PICS/PIcs_Resnet/TrainFile/train_profile.txt') 

dataset_test_sub = CustomDataset('/remote-home/cs_acmis_sdtw/PICS/PIcs_Resnet/TrainFile/test_sub.txt', transform=transform)
dataset_test_time = CustomDataset('/remote-home/cs_acmis_sdtw/PICS/PIcs_Resnet/TrainFile/test_time.txt', transform=transform)
dataset_test_profile = FeatureDataset('/remote-home/cs_acmis_sdtw/PICS/PIcs_Resnet/TrainFile/test_profile.txt')

# 创建DataLoader
train_sub_loader = DataLoader(dataset_train_sub, batch_size=batch_size, shuffle=False)
train_time_loader = DataLoader(dataset_train_time, batch_size=batch_size, shuffle=False)
train_profile_loader = DataLoader(dataset_train_profile, batch_size=batch_size, shuffle=False)

test_sub_loader = DataLoader(dataset_test_sub, batch_size=batch_size, shuffle=False)
test_time_loader = DataLoader(dataset_test_time, batch_size=batch_size, shuffle=False)
test_profile_loader = DataLoader(dataset_test_profile, batch_size=batch_size, shuffle=False)

# 遍历DataLoader  
# for features in train_profile_loader:  
#     print("----------------")
#     print(features.shape)  # 输出特征Tensor的形状
#     print("----------------")

# for i, (images, labels) in enumerate(train_sub_loader):
#     images = images.to(device)
#     labels = labels.to(device)
#     # print(images)
#     print(labels)
#     break

# for i, (images, labels) in enumerate(train_time_loader):
#     images = images.to(device)
#     labels = labels.to(device)
#     # print(images)
#     print(labels)
#     break


# # 从本地文件夹加载训练和测试数据
# train_dataset = torchvision.datasets.ImageFolder(root=r'/remote-home/cs_acmis_sdtw/PICS/PIcs_Resnet\TrainData1', transform=transform)
# test_dataset = torchvision.datasets.ImageFolder(root=r'/remote-home/cs_acmis_sdtw/PICS/PIcs_Resnet\TestData1', transform=transform)
#
# # 数据加载器
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#
class VCCNN(nn.Module):
    def __init__(self):
        super(VCCNN, self).__init__()

        self.DM_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16, padding='valid')
        self.DM_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16, stride=2, padding='valid')
        self.DM_conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, padding='valid')
        self.DM_global_max_pool = nn.AdaptiveMaxPool1d(1)

        self.Time_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 4), stride=(2, 2), padding='valid')
        self.Time_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2), padding='valid')
        self.Time_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding='valid')
        self.Time_global_max_pool = nn.AdaptiveMaxPool2d(1)

        self.Sub_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 4), stride=(2, 2), padding='valid')
        self.Sub_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2), padding='valid')
        self.Sub_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding='valid')
        self.Sub_global_max_pool = nn.AdaptiveMaxPool2d(1)

        # self.Merge_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 16), stride=(2, 4), padding=(1, 0))
        # self.Merge_conv2 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=(3, 16), stride=(2, 4), padding=(1, 0))
        # self.Merge_global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.softmax = nn.Softmax(dim=1)
        self.dense1 = nn.Linear(192, 32)  # 64 * 4 = 256
        self.dense2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, Sub_x,Time_x,DM_x):
        DM_x = F.relu(self.DM_conv1(DM_x))
        # print(DM_x.shape)
        DM_x = F.relu(self.DM_conv2(DM_x))
        # print(DM_x.shape)
        DM_x = F.relu(self.DM_conv3(DM_x))
        # print(DM_x.shape)
        DM_x = self.DM_global_max_pool(DM_x)
        DM_x = DM_x.view(DM_x.size(0), -1)

        Time_x = F.relu(self.Time_conv1(Time_x))
        Time_x = F.relu(self.Time_conv2(Time_x))
        Time_x = F.relu(self.Time_conv3(Time_x))
        Time_x = self.Time_global_max_pool(Time_x)
        Time_x = Time_x.view(Time_x.size(0), -1)

        Sub_x = F.relu(self.Sub_conv1(Sub_x))
        Sub_x = F.relu(self.Sub_conv2(Sub_x))
        Sub_x = F.relu(self.Sub_conv3(Sub_x))
        Sub_x = self.Sub_global_max_pool(Sub_x)
        Sub_x = Sub_x.view(Sub_x.size(0), -1)

        # print(f'{Sub_x.shape}, {Time_x.shape}, {DM_x.shape}')

        # merged_x = torch.cat((Sub_x, Time_x, DM_x), dim=1)
        # merged_x = merged_x.view(merged_x.size(0), 1, 3, 64)
        # print(merged_x.shape)
        #
        # merged_x = F.relu(self.Merge_conv1(merged_x))
        # print(merged_x.shape)
        # merged_x = F.relu(self.Merge_conv2(merged_x))
        # merged_x = self.Merge_global_avg_pool(merged_x)
        # merged_x = merged_x.view(merged_x.size(0), -1)
        # out = self.softmax(merged_x)

        merged_x = torch.cat((Sub_x, Time_x, DM_x), dim=1)
        # print(merged_x.shape)
        # Apply the first dense layer and ReLU activation
        merged_x = F.relu(self.dense1(merged_x))
        # print(merged_x.shape)
        # Apply the second dense layer and sigmoid activation
        out = self.sigmoid(self.dense2(merged_x))
        return out
#
model = VCCNN().to(device)
#
# # subbands_input = torch.randn(1, 1, 64, 64)  # 假设输入大小为 (batch_size, channels, height, width)
# # time_vs_phase_input = torch.randn(1, 1, 64, 64)  # 假设输入大小为 (batch_size, channels, height, width)
# # DM_input = torch.randn(1, 1, 64)
# # output = model(subbands_input, time_vs_phase_input, DM_input)
# # print(output)  # 应该输出 (1, 2)
#
# 损失函数
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.002)
#
#
# 更新学习率
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
#
#
# 训练数据集
total_step = len(train_profile_loader)
print(total_step)
curr_lr = learning_rate

print("----------------------------------训练开始--------------------------------------")
for epoch in range(num_epochs):
    for i, ((sub_images, sub_labels), (time_images, time_labels), profile_data) in enumerate(zip(train_sub_loader, train_time_loader, train_profile_loader)):
  
        # 将数据移动到指定的设备（如 GPU）  
        sub_images = sub_images.to(device)  
        sub_labels = sub_labels.to(device).view(-1, 1).float() 
        time_images = time_images.to(device)  
        time_labels = time_labels.to(device)  
        profile_data = profile_data.to(device)
        # print(sub_labels)

        # print(sub_images.shape)
        # print(profile_data.shape)
        # print(sub_images[0])
        # print(profile_data[0])
        # Forward pass
        # outputs = model(images)
        outputs = model(sub_images, time_images, profile_data)
        # print(outputs)
        loss = criterion(outputs, sub_labels)
        # print(loss)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i + 1) % 100 == 0:
        #     print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
        #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # 延迟学习率
    if (epoch + 1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# # 测试网络模型
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# S将模型保存
torch.save(model.state_dict(), 'HCCNN.ckpt')