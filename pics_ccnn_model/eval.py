import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import csv  

# 判断是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 80  # 80轮
batch_size = 32  # 32步长
learning_rate = 0.01  # 学习率0.01

transform = transforms.Compose([
    transforms.Resize(64),  # 调整图像大小为64x64
    transforms.ToTensor()  # 将图像转换为torch.Tensor
])


# 从本地文件夹加载训练和测试数据
test_dataset = torchvision.datasets.ImageFolder(root=r'/remote-home/cs_acmis_sdtw/PICS/PIcs_Resnet/TrainData2', transform=transform)

data=test_dataset.imgs

# 数据加载器
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

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

# 加载模型权重  
model.load_state_dict(torch.load('resnet_time_2.ckpt'))

# 确保模型处于评估模式  
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    TP = 0  # True Positives  
    FP = 0  # False Positives  
    TN = 0  # True Negatives  
    FN = 0  # False Negatives
    num=0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        for index,result in enumerate(predicted.cpu().numpy()):
            # print(result)
            data[num]=data[num]+(result,)
            num+=1
        
        # 将标签为1的视为正例，0视为负例  
        positive_labels = (labels == 0)  
        negative_labels = (labels == 1)  
        positive_predictions = (predicted == 0)  
        negative_predictions = (predicted == 1)
        # True Positives: 预测为正，实际也为正  
        TP += ((positive_predictions & positive_labels).sum().item())  
        # False Positives: 预测为正，实际为负  
        FP += ((positive_predictions & negative_labels).sum().item())  
        # True Negatives: 预测为负，实际也为负  
        TN += ((negative_predictions & negative_labels).sum().item())  
        # False Negatives: 预测为负，实际为正  
        FN += ((negative_predictions & positive_labels).sum().item())

        # print(f'label:{labels},  pre:{predicted}')
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(total)
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    print(f'True Positives: {TP}')  
    print(f'False Positives: {FP}')  
    print(f'True Negatives: {TN}')  
    print(f'False Negatives: {FN}')   
    print(total)
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

# 指定CSV文件的名称  
filename = "/remote-home/cs_acmis_sdtw/PICS/train_time2.csv"  
# 使用'with'语句打开文件，确保正确关闭文件  
with open(filename, 'w', newline='', encoding='utf-8') as csvfile:  
    # 创建一个csv写入器  
    writer = csv.writer(csvfile)  
    # 写入数据  
    writer.writerows(data)  