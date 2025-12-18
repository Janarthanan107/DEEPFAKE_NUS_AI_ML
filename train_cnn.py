#!/usr/bin/env python3
"""
CNN + LSTM Training Script for Deepfake Detection

This script trains a CNN+LSTM model to detect deepfakes in videos.
Best for: Videos with motion and temporal patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

class VideoDeepfakeDataset(Dataset):
    """Dataset for deepfake videos."""
    
    def __init__(self, data_dir, transform=None, num_frames=16, split='train'):
        """
        Args:
            data_dir: Directory with 'real' and 'fake' subdirectories
            transform: Optional transform to apply
            num_frames: Number of frames to sample per video
            split: 'train' or 'val'
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.num_frames = num_frames
        self.samples = []
        
        # Load real videos (label = 0)
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for vid_path in real_dir.glob('*.mp4') + real_dir.glob('*.avi'):
                self.samples.append((str(vid_path), 0))
        
        # Load fake videos (label = 1)
        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for vid_path in fake_dir.glob('*.mp4') + fake_dir.glob('*.avi'):
                self.samples.append((str(vid_path), 1))
        
        print(f"Loaded {len(self.samples)} {split} video samples")
        
    def __len__(self):
        return len(self.samples)
    
    def sample_frames(self, video_path, num_frames):
        """Sample frames uniformly from video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames:
            # If video has fewer frames, repeat frames
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            # Sample uniformly
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
        
        cap.release()
        
        # If we didn't get enough frames, pad with the last frame
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else torch.zeros(3, 224, 224))
        
        return torch.stack(frames)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        try:
            frames = self.sample_frames(video_path, self.num_frames)
            return frames, label
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # Return a random different sample
            return self.__getitem__((idx + 1) % len(self))

class CNNLSTMDeepfakeDetector(nn.Module):
    """CNN + LSTM for video deepfake detection."""
    
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2):
        super().__init__()
        
        # CNN backbone (ResNet-18)
        resnet = models.resnet18(pretrained=True)
        # Remove the final FC layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512,  # ResNet-18 output
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, num_frames, channels, height, width)
        batch_size, num_frames, c, h, w = x.size()
        
        # Extract features from each frame
        # Reshape to (batch * num_frames, channels, height, width)
        x = x.view(batch_size * num_frames, c, h, w)
        
        # Pass through CNN
        features = self.cnn(x)  # (batch * num_frames, 512, 1, 1)
        features = features.view(batch_size, num_frames, -1)  # (batch, num_frames, 512)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # (batch, num_frames, hidden_size)
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Classify
        output = self.fc(last_output)  # (batch, num_classes)
        
        return output

def get_transforms(img_size=224, augment=True):
    """Get video frame transforms."""
    
    if augment:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
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
    for frames, labels in pbar:
        frames, labels = frames.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
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
        for frames, labels in pbar:
            frames, labels = frames.to(device), labels.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
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
    parser = argparse.ArgumentParser(description='Train CNN+LSTM for Deepfake Detection')
    parser.add_argument('--data_dir', type=str, default='datasets/videos', help='Data directory')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames per video')
    parser.add_argument('--img_size', type=int, default=224, help='Frame size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--output', type=str, default='cnn_lstm_deepfake.pth', help='Output model path')
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
    print(f"🚀 Training CNN+LSTM Deepfake Detector")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Frames per video: {args.num_frames}")
    print(f"Frame size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*70}\n")
    
    # Create datasets
    train_transform = get_transforms(args.img_size, augment=True)
    val_transform = get_transforms(args.img_size, augment=False)
    
    train_dataset = VideoDeepfakeDataset(args.data_dir, transform=train_transform, 
                                         num_frames=args.num_frames, split='train')
    val_dataset = VideoDeepfakeDataset(args.data_dir, transform=val_transform, 
                                       num_frames=args.num_frames, split='val')
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        print(f"Expected structure: {args.data_dir}/real/ and {args.data_dir}/fake/")
        return
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create model
    print("📦 Loading model...")
    model = CNNLSTMDeepfakeDetector(num_classes=2, hidden_size=args.hidden_size, 
                                    num_layers=args.num_layers)
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
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, precision, recall, f1, auc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
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
