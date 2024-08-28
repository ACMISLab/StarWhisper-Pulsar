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

# 加载模型权重  
model.load_state_dict(torch.load('HCCNN.ckpt'))

# 确保模型处于评估模式  
model.eval()

print("----------------------------------评估开始--------------------------------------")
with torch.no_grad():
    TP = 0  # True Positives  
    FP = 0  # False Positives  
    TN = 0  # True Negatives  
    FN = 0  # False Negatives
    num=0
    for i, ((sub_images, sub_labels), (time_images, time_labels), profile_data) in enumerate(zip(test_sub_loader, test_time_loader, test_profile_loader)):
        # 将数据移动到指定的设备（如 GPU）  
        sub_images = sub_images.to(device)  
        sub_labels = sub_labels.to(device)
        time_images = time_images.to(device)  
        time_labels = time_labels.to(device)  
        profile_data = profile_data.to(device)

        outputs = model(sub_images, time_images, profile_data)
        outputs = outputs.squeeze(1)
        predicted = (outputs > 0.5).float()  # 将输出转换为0或1

        # 计算TP, FP, TN, FN  
        TP += ((predicted == 0) & (sub_labels == 0)).sum().item()  
        FP += ((predicted == 0) & (sub_labels == 1)).sum().item()  
        TN += ((predicted == 1) & (sub_labels == 1)).sum().item()  
        FN += ((predicted == 1) & (sub_labels == 0)).sum().item() 

    print(f'True Positives: {TP}')  
    print(f'False Positives: {FP}')  
    print(f'True Negatives: {TN}')  
    print(f'False Negatives: {FN}')   
    print(f'total: {(TP+FP+TN+FN)}')
    # 计算准确率
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    print(f"Accuracy: {accuracy}")

    # 计算召回率
    recall = TP / (TP + FN)
    print(f"Recall: {recall}")

    # 计算精确率
    precision = TP / (TP + FP)
    print(f"Precision: {precision}")

    # 计算 F1 分数
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"F1-Score: {f1_score} {2*TP/(2*TP+FP+FN)}")