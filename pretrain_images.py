import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import time
import matplotlib.pyplot as plt  
from src.models import BasicConvClassifier
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau

print("Current working directory:", os.getcwd())

class CustomImageDataset(Dataset):
    #def __init__(self, img_dir, img_paths,transform=None):
    def __init__(self, img_dir, img_paths, labels, transform=None):
        self.img_dir = img_dir
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

#def load_image_paths(txt_file):
#    with open(txt_file, 'r') as f:
#        image_paths = f.read().splitlines()
#    return image_paths
def load_image_paths_and_labels(txt_file):
    with open(txt_file, 'r') as f:
        image_paths = f.read().splitlines()
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]  # 生成标签
    print("Extracted Labels:", labels[:10])  # 打印部分标签以验证正确性
    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}  # 创建标签到索引的映射
    print("Label to Index Mapping:", label_to_idx)  # 打印标签到索引的映射
    labels = [label_to_idx[label] for label in labels]  # 将标签转换为索引
    print("Indexed Labels:", labels[:10])  # 打印部分索引化标签以验证正确性
    return image_paths, labels, label_to_idx

#train_image_paths = load_image_paths('/content/drive/MyDrive/DL/最終課題/dl_lecture_competition_pub-MEG-competition/data/fixed_train_image_paths.txt')
#val_image_paths = load_image_paths('/content/drive/MyDrive/DL/最終課題/dl_lecture_competition_pub-MEG-competition/data/fixed_val_image_paths.txt')
train_image_paths, train_labels, label_to_idx = load_image_paths_and_labels('/content/drive/MyDrive/DL/最終課題/dl_lecture_competition_pub-MEG-competition/data/fixed_train_image_paths.txt')
val_image_paths, val_labels, _ = load_image_paths_and_labels('/content/drive/MyDrive/DL/最終課題/dl_lecture_competition_pub-MEG-competition/data/fixed_val_image_paths.txt')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    Rearrange('c h w -> c (h w)')  # 将图像从 (C, H, W) 转换为 (C, H * W)
])

print("Loading train set...")
#train_dataset = CustomImageDataset(img_dir='/content/drive/MyDrive/DL/最終課題/dl_lecture_competition_pub-MEG-competition/data/Images', img_paths=train_image_paths, transform=transform)
train_dataset = CustomImageDataset(img_dir='/content/drive/MyDrive/DL/最終課題/dl_lecture_competition_pub-MEG-competition/data/Images', img_paths=train_image_paths, labels=train_labels, transform=transform)
#train_subset = Subset(train_dataset, range(0, len(train_dataset), 10))  # 使用子集进行训练
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)  # 增加 num_workers

print("Loading val set...")
#val_dataset = CustomImageDataset(img_dir='/content/drive/MyDrive/DL/最終課題/dl_lecture_competition_pub-MEG-competition/data/Images', img_paths=val_image_paths, transform=transform)
val_dataset = CustomImageDataset(img_dir='/content/drive/MyDrive/DL/最終課題/dl_lecture_competition_pub-MEG-competition/data/Images', img_paths=val_image_paths, labels=val_labels, transform=transform)
#val_subset = Subset(val_dataset, range(0, len(val_dataset), 10))  # 使用子集进行验证
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)  # 增加 num_workers

# 加载预训练模型（如ResNet）
#model = models.resnet50(pretrained=True)
# 加载预训练模型 BasicConvClassifier
num_classes = len(label_to_idx)
num_subjects = 4  # Dummy value for num_subjects as it's not used in pretraining
model = BasicConvClassifier(num_classes=num_classes, seq_len=224, in_channels=3, num_subjects=num_subjects)

# 修改最后一层以适应我们的任务
#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, 1854)  # 1854 是类别数
#model.fc = nn.Linear(num_ftrs, len(label_to_idx))  # 使用标签数作为输出

# 冻结预训练模型的大部分层，只训练最后几层
#for param in list(model.parameters())[:-10]:
#    param.requires_grad = False

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)  # 调整学习率

# 定义学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# 训练模型
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    #for inputs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        #inputs = inputs.to(device)
        #labels = torch.randint(0, 1854, (inputs.size(0),)).to(device)  # 生成随机标签
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        #outputs = model(inputs)
        outputs = model(inputs, torch.zeros(inputs.size(0), dtype=torch.long).to(device))  # Dummy subject_idxs
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    end_time = time.time()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {end_time - start_time:.2f} seconds')
    
    # 验证步骤
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, torch.zeros(inputs.size(0), dtype=torch.long).to(device))  # Dummy subject_idxs
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}')
    
    # 学习率调度器步骤
    scheduler.step(val_loss)

# 保存预训练模型的权重
torch.save(model.state_dict(), 'pretrained_model.pth')
