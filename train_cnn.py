#!/usr/bin/env python3
"""
CNN + LSTM Training Script for Video Deepfake Detection

This script trains a temporal model (ResNet + LSTM) on sequences of video frames.
Best for: Detecting dynamic/temporal anomalies in videos.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import glob
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

class VideoFrameDataset(Dataset):
    """Dataset that groups frames into video sequences."""
    
    def __init__(self, data_dir, sequence_length=10, transform=None, split='train', val_split=0.2):
        self.data_dir = Path(data_dir)
        self.seq_len = sequence_length
        self.transform = transform
        
        # Group frames by video ID
        self.video_frames = defaultdict(list)
        self.labels = {}
        
        # Helper to process directory
        def process_dir(name, label):
            dir_path = self.data_dir / name
            if not dir_path.exists():
                return
            
            # Find all images
            images = list(dir_path.glob('*.png')) + list(dir_path.glob('*.jpg'))
            
            for img in images:
                # Assuming format: videoID_frameNum.png
                filename = img.name
                video_id = filename.rsplit('_', 1)[0]
                self.video_frames[video_id].append(str(img))
                self.labels[video_id] = label
        
        process_dir('real', 0)
        process_dir('fake', 1)
        
        # Sort frames for each video to ensure temporal order
        for vid in self.video_frames:
            self.video_frames[vid].sort()
            
        # Get all video IDs
        all_videos = list(self.video_frames.keys())
        train_vids, val_vids = train_test_split(all_videos, test_size=val_split, random_state=42)
        
        self.video_ids = train_vids if split == 'train' else val_vids
        print(f"Loaded {len(self.video_ids)} {split} videos (Total frames: {sum(len(self.video_frames[v]) for v in self.video_ids)})")

    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frames = self.video_frames[video_id]
        label = self.labels[video_id]
        
        # improved sampling: take evenly spaced frames
        frame_indices = np.linspace(0, len(frames)-1, self.seq_len, dtype=int)
        selected_frames = [frames[i] for i in frame_indices]
        
        images = []
        for p in selected_frames:
            try:
                img = Image.open(p).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except:
                # Handle corrupted images by duplicating previous or zero
                if len(images) > 0:
                    images.append(images[-1])
                else:
                    images.append(torch.zeros(3, 224, 224))
        
        # Stack into [Seq_Len, C, H, W]
        images = torch.stack(images)
        return images, label

class ResNetLSTM(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=256, num_layers=2):
        super(ResNetLSTM, self).__init__()
        
        # CNN Encoder (ResNet18)
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1]) # Remove fc layer
        self.cnn_out_dim = resnet.fc.in_features
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Classifier
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: [Batch, Seq, C, H, W]
        batch_size, seq_len, c, h, w = x.size()
        
        # Flatten for CNN: [Batch * Seq, C, H, W]
        c_in = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features
        features = self.cnn(c_in) # [Batch * Seq, 512, 1, 1]
        features = features.view(batch_size, seq_len, -1) # [Batch, Seq, 512]
        
        # LSTM
        lstm_out, _ = self.lstm(features) # [Batch, Seq, Hidden]
        
        # Take last time step
        last_out = lstm_out[:, -1, :]
        
        # Classify
        out = self.fc(last_out)
        return out

def train(args):
    # Device config
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Data
    print("Preparing data...")
    train_dataset = VideoFrameDataset(args.data_dir, args.seq_len, transform=transform, split='train')
    val_dataset = VideoFrameDataset(args.data_dir, args.seq_len, transform=transform, split='val')
    
    if len(train_dataset) == 0:
        print("❌ No data found! Check path.")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Model
    model = ResNetLSTM().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss/len(train_loader), 'acc': 100*correct/total})
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        print(f"Validation Acc: {100*val_correct/val_total:.2f}%")
        
        # Save model
        torch.save(model.state_dict(), args.output)
        
    print(f"✅ Training complete. Model saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/videos')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8) # Smaller batch for video sequences (memory intensive)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output', type=str, default='cnn_lstm_deepfake.pth')
    
    args = parser.parse_args()
    train(args)
