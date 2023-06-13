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
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  
])

def vertical_diameter(binary_segmentation):
    '''
    Get the vertical diameter from a binary segmentation.
    The vertical diameter is defined as the "fattest" area of the binary_segmentation parameter.
    '''
    if binary_segmentation.ndim == 1:
        binary_segmentation = binary_segmentation[None, :]

    vertical_axis_diameter = np.sum(binary_segmentation, axis=1)

    if vertical_axis_diameter.ndim > 1:
        diameter = np.max(vertical_axis_diameter, axis=1)
    else:
        diameter = np.max(vertical_axis_diameter)

    return diameter


def vertical_cup_to_disc_ratio(od, oc):
    '''
    Compute the vertical cup-to-disc ratio from a given labelling map.
    '''
    cup_diameter = vertical_diameter(oc)
    disc_diameter = vertical_diameter(od)

    return cup_diameter / (disc_diameter + 0.0000001)

def calculate_cdr(image_path, disc_model, cup_model, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    disc_pred = disc_model(image)
    cup_pred = cup_model(image)

    disc_mask = (disc_pred.argmax(dim=1) > 0.5).squeeze().cpu().numpy()
    cup_mask = (cup_pred.argmax(dim=1) > 0.5).squeeze().cpu().numpy()

    cdr = vertical_cup_to_disc_ratio(disc_mask, cup_mask)

    return cdr
