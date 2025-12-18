#!/usr/bin/env python3
"""
Deepfake Detection Inference Script
Combines Gating Classifier, ViT (Image), and CNN-LSTM (Video) models.
"""

import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import timm
import joblib
import argparse
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import our modules (assuming they are in the same directory)
try:
    from feature_extraction import extract_video_features
    from gating_model import predict_with_confidence
except ImportError:
    print("Error: Could not import helper modules. Run from project root.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Model Definitions (must match training scripts)
# -----------------------------------------------------------------------------

class ViTDeepfakeDetector(nn.Module):
    """Vision Transformer for deepfake detection."""
    def __init__(self, model_name='vit_base_patch16_224', pretrained=False, num_classes=2):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        return self.vit(x)

class ResNetLSTM(nn.Module):
    """CNN + LSTM for video deepfake detection."""
    def __init__(self, num_classes=2, hidden_dim=256, num_layers=2):
        super(ResNetLSTM, self).__init__()
        resnet = models.resnet18(weights=None) # No weights needed for inference load
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_out_dim = resnet.fc.in_features
        self.lstm = nn.LSTM(input_size=self.cnn_out_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        c_in = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(c_in)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out

# -----------------------------------------------------------------------------
# Inference Logic
# -----------------------------------------------------------------------------

class DeepfakeDetector:
    def __init__(self, device='auto'):
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        print(f"🚀 Initializing Deepfake Detector on {self.device}...")
        
        # Paths
        self.gating_path = "gating_rf.joblib"
        self.vit_path = "vit_deepfake.pth"
        self.cnn_path = "cnn_lstm_deepfake.pth"
        
        # Load Gating Model
        if os.path.exists(self.gating_path):
            self.gating_clf = joblib.load(self.gating_path)
            print("✅ Loaded Gating Classifier")
        else:
            print("❌ Gating model not found!")
            sys.exit(1)
            
        # Load ViT Model
        self.vit_model = None
        if os.path.exists(self.vit_path):
            try:
                # Initialize architecture
                self.vit_model = ViTDeepfakeDetector()
                # Load weights (handle incomplete training or different save formats)
                checkpoint = torch.load(self.vit_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.vit_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.vit_model.load_state_dict(checkpoint)
                
                self.vit_model.to(self.device).eval()
                print("✅ Loaded ViT Model")
            except Exception as e:
                print(f"⚠️ Failed to load ViT model: {e}")
        else:
            print("⚠️ ViT model file not found (training might be in progress)")

        # Load CNN Model
        self.cnn_model = None
        if os.path.exists(self.cnn_path):
            try:
                self.cnn_model = ResNetLSTM()
                self.cnn_model.load_state_dict(torch.load(self.cnn_path, map_location=self.device))
                self.cnn_model.to(self.device).eval()
                print("✅ Loaded CNN-LSTM Model")
            except Exception as e:
                print(f"⚠️ Failed to load CNN model: {e}")
        else:
            print("⚠️ CNN model file not found")
            
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_vit(self, video_path, num_frames=5):
        """Run ViT on sampled frames from the video."""
        if not self.vit_model:
            return None, 0.0
            
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Randomly sample frames
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(self.transform(frame))
                if len(frames) >= num_frames: break
        cap.release()
        
        if not frames:
            return None, 0.0
            
        # Batch inference
        batch = torch.stack(frames).to(self.device)
        with torch.no_grad():
            outputs = self.vit_model(batch)
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of Fake
            avg_prob = torch.mean(probs).item()
            
        label = "FAKE" if avg_prob > 0.5 else "REAL"
        return label, avg_prob

    def predict_cnn(self, video_path, seq_len=10):
        """Run CNN-LSTM on a sequence of frames."""
        if not self.cnn_model:
            return None, 0.0
            
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample sequence
        indices = np.linspace(0, total_frames-1, seq_len, dtype=int)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(self.transform(frame))
        cap.release()
        
        # Pad if not enough frames
        while len(frames) < seq_len:
            frames.append(torch.zeros(3, 224, 224))
            
        frames = frames[:seq_len]
        
        # Create batch [1, Seq, C, H, W]
        batch = torch.stack(frames).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.cnn_model(batch)
            prob = torch.softmax(output, dim=1)[0, 1].item()
            
        label = "FAKE" if prob > 0.5 else "REAL"
        return label, prob

    def analyze_video(self, video_path):
        """Full pipeline: Extract features -> Route -> Predict."""
        print(f"\n🔍 Analyzing: {video_path}")
        
        # 1. Extract Features
        features = extract_video_features(video_path)
        if features is None:
            return {"error": "Could not process video"}
            
        # 2. Gating Classifier
        route_pred = self.gating_clf.predict([features])[0]
        route_class = {0: "ViT", 1: "CNN", 2: "ViT + CNN"}.get(route_pred, "ViT + CNN")
        
        print(f"🛣️ Routing Decision: {route_class}")
        
        results = {
            "routing": route_class,
            "final_prediction": "UNKNOWN",
            "confidence": 0.0,
            "details": {}
        }
        
        vit_score = None
        cnn_score = None
        
        # 3. Execute Models based on Route
        if "ViT" in route_class and self.vit_model:
            label, score = self.predict_vit(video_path)
            vit_score = score
            results["details"]["ViT"] = {"label": label, "score": score}
            print(f"  📸 ViT Prediction: {label} ({score:.4f})")
            
        if "CNN" in route_class and self.cnn_model:
            label, score = self.predict_cnn(video_path)
            cnn_score = score
            results["details"]["CNN"] = {"label": label, "score": score}
            print(f"  🎥 CNN Prediction: {label} ({score:.4f})")
            
        # 4. Ensemble Logic (Simple Average)
        scores = []
        if vit_score is not None: scores.append(vit_score)
        if cnn_score is not None: scores.append(cnn_score)
        
        if scores:
            avg_score = sum(scores) / len(scores)
            results["confidence"] = avg_score
            results["final_prediction"] = "FAKE" if avg_score > 0.5 else "REAL"
        else:
            results["final_prediction"] = "ERROR"
            results["error"] = "No models available for routing decision"
            
        print(f"✨ FINAL VERDICT: {results['final_prediction']} ({results['confidence']:.2%})\n")
        return results

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print("❌ Video file not found!")
        return
        
    detector = DeepfakeDetector()
    detector.analyze_video(args.video)

if __name__ == "__main__":
    main()
