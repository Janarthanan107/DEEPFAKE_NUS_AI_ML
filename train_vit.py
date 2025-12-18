#!/usr/bin/env python3
"""
Vision Transformer (ViT) Training Script for Deepfake Detection

This script trains a ViT model to detect deepfakes in images.
Best for: High-resolution, static images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import timm
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

class DeepfakeDataset(Dataset):
    """Dataset for deepfake images."""
    
    def __init__(self, data_dir, transform=None, split='train'):
        """
        Args:
            data_dir: Directory with 'real' and 'fake' subdirectories
            transform: Optional transform to apply
            split: 'train' or 'val'
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Load real images (label = 0)
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for img_path in list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png')):
                self.samples.append((str(img_path), 0))
        
        # Load fake images (label = 1)
        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for img_path in list(fake_dir.glob('*.jpg')) + list(fake_dir.glob('*.png')):
                self.samples.append((str(img_path), 1))
        
        print(f"Loaded {len(self.samples)} {split} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random different sample
            return self.__getitem__((idx + 1) % len(self))

class ViTDeepfakeDetector(nn.Module):
    """Vision Transformer for deepfake detection."""
    
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, num_classes=2):
        super().__init__()
        # Load pretrained ViT from timm
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        return self.vit(x)

def get_transforms(img_size=224, augment=True):
    """Get image transforms for training and validation."""
    
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of being fake
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    
    return avg_loss, accuracy, precision, recall, f1, auc

def main():
    parser = argparse.ArgumentParser(description='Train ViT for Deepfake Detection')
    parser.add_argument('--data_dir', type=str, default='datasets/images', help='Data directory')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', help='ViT model variant')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output', type=str, default='vit_deepfake.pth', help='Output model path')
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda, mps, cpu, or auto')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*70}")
    print(f"🚀 Training ViT Deepfake Detector")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*70}\n")
    
    # Create datasets
    train_transform = get_transforms(args.img_size, augment=True)
    val_transform = get_transforms(args.img_size, augment=False)
    
    train_dataset = DeepfakeDataset(args.data_dir, transform=train_transform, split='train')
    val_dataset = DeepfakeDataset(args.data_dir, transform=val_transform, split='val')
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        print(f"Expected structure: {args.data_dir}/real/ and {args.data_dir}/fake/")
        return
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    print("📦 Loading model...")
    model = ViTDeepfakeDetector(model_name=args.model_name, pretrained=True, num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_auc = 0.0
    print("\n🔧 Starting training...\n")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, precision, recall, f1, auc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        # Save best model
        if auc > best_auc:
            best_auc = auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': auc,
                'accuracy': val_acc,
            }, args.output)
            print(f"  ✅ Saved best model (AUC: {auc:.4f})")
        
        print()
    
    print(f"\n{'='*70}")
    print(f"✅ Training Complete!")
    print(f"{'='*70}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Model saved to: {args.output}")
    print()

if __name__ == '__main__':
    main()
