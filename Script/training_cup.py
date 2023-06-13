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
import cv2
import segmentation_models_pytorch as sm

def create_mask_from_txt(txt_path, img_size, padding=10):
    with open(txt_path, 'r') as f:
        coords = list(map(float, f.read().split()))[1:]
    coords = np.array(coords).reshape(-1, 2)
    coords[:, 0] *= img_size[1]
    coords[:, 1] *= img_size[0]

    coords[:, 0] = np.clip(coords[:, 0] + padding, 0, img_size[1] - 1)
    coords[:, 1] = np.clip(coords[:, 1] + padding, 0, img_size[0] - 1)

    mask = np.zeros(img_size, dtype=np.uint8)

    cv2.fillPoly(mask, [coords.astype(int)], color=1)

    return mask

mask_dir = '/content/cup_labels'
txt_files = os.listdir(mask_dir)
mask_save_dir = '/content/cup_mask'
os.makedirs(mask_save_dir, exist_ok=True)

for txt_file in txt_files:
    txt_path = os.path.join(mask_dir, txt_file)
    mask = create_mask_from_txt(txt_path, img_size=(512, 512))
    mask_path = os.path.join(mask_save_dir, txt_file.replace('.txt', '.png'))
    cv2.imwrite(mask_path, mask * 255)

csv_file_path = '/content/drive/MyDrive/siriraj-eye-dataset/all_labels_processed.csv'
df = pd.read_csv(csv_file_path)

df = df.drop(columns=['camera','glaucoma_suspect','image_quality','image_quality_split', 'has_cup', 'has_disc', 'filename', 'has_disc_split','dr_label','dr_label_split'])

old_base_path = '../dataset/siriraj-eye-dataset-2023-jan/all_images'
new_base_path = '/content/all_images'
df['path'] = df['path'].apply(lambda x: x.replace(old_base_path, new_base_path).replace('\\', '/'))

df.set_index('path', inplace=False)

df = df.dropna(subset=['has_cup_split'])

train_df = df[df['has_cup_split'] == 'train']
test_df = df[df['has_cup_split'] == 'test']
train_df, val_df = train_test_split(train_df, test_size=652/len(train_df), random_state=42)
train_df.drop(columns=['has_cup_split'], inplace=True)
val_df.drop(columns=['has_cup_split'], inplace=True)
test_df.drop(columns=['has_cup_split'], inplace=True)

class EyeDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

train_images = train_df['path'].tolist()
train_masks = [path.replace('all_images', 'cup_mask').replace('.jpg', '.png') for path in train_images]

val_images = val_df['path'].tolist()
val_masks = [path.replace('all_images', 'cup_mask').replace('.jpg', '.png') for path in val_images]

test_images = test_df['path'].tolist()
test_masks = [path.replace('all_images', 'cup_mask').replace('.jpg', '.png') for path in test_images]

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

train_dataset = EyeDataset(train_images, train_masks, transform=transform)
val_dataset = EyeDataset(val_images, val_masks, transform=transform)
test_dataset = EyeDataset(test_images, test_masks, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = sm.Unet('resnet34', classes=2, activation=None)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def iou_score(output, target):
    output = torch.argmax(output, dim=1)
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = intersection / union
    return iou.item()

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).long()
        masks = masks.squeeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_iou += iou_score(outputs, masks)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_iou = running_iou / len(loader.dataset)

    return epoch_loss, epoch_iou


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device).long()
            masks = masks.squeeze(1)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)
            running_iou += iou_score(outputs, masks)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_iou = running_iou / len(loader.dataset)

    return epoch_loss, epoch_iou

BATCH_SIZE = 32
IMG_SIZE = (512, 512)
EPOCHS = 20

best_loss = 10
best_model_path = '/content/drive/MyDrive/siriraj-eye-dataset/cup_model.pth'
patience = 5
patience_counter = 0

for epoch in range(EPOCHS):
    train_loss, train_iou = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_iou = validate(model, val_loader, criterion, device)
    print(f'Epoch [{epoch+1}/{EPOCHS}], '
          f'Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')

    if val_loss < best_loss:
        print(f'New best model found! Previous best loss: {best_loss:.4f}, new best loss: {val_loss:.4f}. Saving model...')
        torch.save(model.state_dict(), best_model_path)
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f'Patience counter: {patience_counter}')
        if patience_counter >= patience:
            print('Early stopping triggered.')
            break
