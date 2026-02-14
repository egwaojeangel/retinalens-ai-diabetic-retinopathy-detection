import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm

# ────────────────────────────────────────────────
#  CONFIG – change only here if needed
# ────────────────────────────────────────────────
SAVE_PATH = "best_model.pth"           # will be created here
BASE_PATH = r"C:\Users\egwao\Downloads\aptos2019-blindness-detection"
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-4

train_csv_path    = os.path.join(BASE_PATH, "train.csv")
train_images_path = os.path.join(BASE_PATH, "train_images")

# ────────────────────────────────────────────────
#  Load & prepare data
# ────────────────────────────────────────────────
df = pd.read_csv(train_csv_path)
df["diagnosis"] = df["diagnosis"].apply(lambda x: 0 if x == 0 else 1)

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["diagnosis"], random_state=42
)

print(f"Train samples: {len(train_df):,}   Val samples: {len(val_df):,}")

# ────────────────────────────────────────────────
#  Transforms & Dataset
# ────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class APTOSDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self): return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.loc[idx, "id_code"] + ".png"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.loc[idx, "diagnosis"]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

train_dataset = APTOSDataset(train_df, train_images_path, train_transforms)
val_dataset   = APTOSDataset(val_df,   train_images_path, val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ────────────────────────────────────────────────
#  Model
# ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = timm.create_model("efficientnet_b0", pretrained=True)
for param in model.parameters():
    param.requires_grad = False
for param in model.blocks[-2:].parameters():
    param.requires_grad = True

model.classifier = nn.Linear(model.classifier.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# ────────────────────────────────────────────────
#  Training functions
# ────────────────────────────────────────────────
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc   = (all_preds == all_labels).mean()
    prec  = precision_score(all_labels, all_preds, zero_division=0)
    rec   = recall_score(all_labels, all_preds, zero_division=0)
    f1    = f1_score(all_labels, all_preds, zero_division=0)
    auc   = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return acc, prec, rec, f1, auc

# ────────────────────────────────────────────────
#  Training loop
# ────────────────────────────────────────────────
best_auc = -1.0
best_epoch = -1

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader)
    val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(model, val_loader)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val AUC: {val_auc:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  → New best AUC! Saved model to: {os.path.abspath(SAVE_PATH)}")

print("\n" + "═"*70)
print(f"Training finished. Best AUC = {best_auc:.4f} (epoch {best_epoch})")
print(f"Final model file: {os.path.abspath(SAVE_PATH)}")
print("═"*70)

# Verify file exists
if os.path.exists(SAVE_PATH):
    print("SUCCESS: best_model.pth was created.")
    print("You can now copy/move it next to webapp.py")
else:
    print("!!! WARNING: best_model.pth NOT found after training !!!")

