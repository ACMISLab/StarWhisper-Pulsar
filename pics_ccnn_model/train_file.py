import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random

transform = transforms.Compose([
    transforms.Resize(64),  # 调整图像大小为64x64
    transforms.ToTensor()  # 将图像转换为torch.Tensor
])

# 从本地文件夹加载训练和测试数据
train_dataset = torchvision.datasets.ImageFolder(root=r'/remote-home/cs_acmis_sdtw/PICS/PIcs_Resnet/TrainData1', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root=r'/remote-home/cs_acmis_sdtw/PICS/PIcs_Resnet/TestData1', transform=transform)

train=[]
for i in train_dataset.imgs:
    train.append(i)
random.shuffle(train)
print(train)
# # 数据加载器
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


for data in train:
    with open('train.txt', 'a') as file:  
        file.write(f'{data[0]} {data[1]}\n')

for data in test_dataset.imgs:
    with open('test.txt', 'a') as file:  
        file.write(f'{data[0]} {data[1]}\n')
