import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from PIL import UnidentifiedImageError

csv_file_path = '/content/drive/MyDrive/siriraj-eye-dataset/all_labels_processed.csv'
df = pd.read_csv(csv_file_path)

df = df.drop(columns=['dr_label','camera','glaucoma_suspect', 'has_cup', 'has_disc', 'filename', 'dr_label_split', 'has_cup_split', 'has_disc_split'])

old_base_path = '../dataset/siriraj-eye-dataset-2023-jan/all_images'
new_base_path = '/content/all_images'
df['path'] = df['path'].apply(lambda x: x.replace(old_base_path, new_base_path).replace('\\', '/'))

df.set_index('path', inplace=False)

df = df.dropna(subset=['image_quality_split'])
encoder = LabelEncoder()
df['image_quality_encoded'] = encoder.fit_transform(df['image_quality'])

train_df = df[df['image_quality_split'] == 'train']
train_df, val_df = train_test_split(train_df, test_size=729, random_state=42, stratify=train_df['image_quality_encoded'])
test_df = df[df['image_quality_split'] == 'test']
train_df.drop(columns=['image_quality', 'image_quality_split'], inplace=True)
val_df.drop(columns=['image_quality', 'image_quality_split'], inplace=True)
test_df.drop(columns=['image_quality', 'image_quality_split'], inplace=True)

BATCH_SIZE = 32
IMG_SIZE = (512, 512)
EPOCHS = 20

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
      img_path, label = self.dataframe.iloc[index]
      try:
        image = Image.open(img_path).convert('RGB')
      except UnidentifiedImageError:
        print(f"Skipping corrupted or unsupported image: {img_path}")
        image = Image.new('RGB', (512, 512), color='white')
      image = self.transform(image)
      return image, label

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = val_transform

train_dataset = CustomDataset(train_df, transform=train_transform)
val_dataset = CustomDataset(val_df, transform=val_transform)
test_dataset = CustomDataset(test_df, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet50(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(torch.log(outputs), labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(data_loader), correct / total

def validate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)
            loss = criterion(torch.log(outputs), labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(data_loader), correct / total

best_val_loss = 1.0
best_model_path = '/content/drive/MyDrive/siriraj-eye-dataset/quality_model.pth'
patience = 5
wait = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f'Epoch [{epoch+1}/{EPOCHS}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    if val_loss < best_val_loss:
        print(f'New best model found! Previous best: {best_val_loss:.4f}, new best: {val_loss:.4f}. Saving model...')
        torch.save(model.state_dict(), best_model_path)
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1

    if wait > patience:
        print(f'Early stopping after {epoch+1} epochs')
        break
