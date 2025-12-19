# 🚀 Complete Setup Guide - Deepfake Detection System

## 📊 Current Status

### ✅ What's Already Done:
1. **Gating Classifier (Router)** - TRAINED ✅
   - Trained with synthetic data
   - Model saved: `gating_rf.joblib`
   - Can route videos to ViT, CNN, or Both

### ❌ What's NOT Done Yet:
1. **ViT Model** - Needs training on real deepfake images
2. **CNN Model** - Needs training on real deepfake videos
3. **Real Dataset** - Need to download from Kaggle

---

## 🎯 Complete Workflow

```
Step 1: Download Dataset (NOW) ←─ You are here
    ↓
Step 2: Train ViT Model
    ↓
Step 3: Train CNN Model  
    ↓
Step 4: (Optional) Retrain Gating Classifier with real videos
    ↓
Step 5: Build Ensemble System
    ↓
Step 6: Deploy!
```

---

## 📥 STEP 1: Download Datasets from Kaggle

### Option A: Quick Setup (Recommended)

```bash
# 1. Run the setup script (will guide you through)
./setup_kaggle.sh

# 2. Select and download a dataset
python3 download_datasets.py
```

### Option B: Manual Setup

#### 1️⃣ Get Kaggle API Token
1. Go to: https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token"
4. Download `kaggle.json`

#### 2️⃣ Install the Token
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 3️⃣ Test It
```bash
kaggle datasets list --max-size 1000000 | head -5
```

#### 4️⃣ Download a Dataset
```bash
# Option 1: Small dataset (600MB) - Good for testing
kaggle datasets download -d manjilkarki/deepfake-and-real-images -p datasets --unzip

# Option 2: Large dataset (5GB) - Better for production
kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p datasets --unzip

# Option 3: Video dataset (4GB)
# First accept the competition rules at: https://www.kaggle.com/c/deepfake-detection-challenge
kaggle competitions download -c deepfake-detection-challenge -p datasets
```

---

## 🔬 STEP 2: Train ViT Model (After Dataset Download)

Once you have images, train the ViT model:

```bash
# Coming soon - will create train_vit.py
# This will train Vision Transformer on images
python3 train_vit.py --data_dir datasets/images --epochs 50
```

---

## 🎥 STEP 3: Train CNN Model

For temporal/video analysis:

```bash
# Coming soon - will create train_cnn.py
# This will train CNN+LSTM on video frames
python3 train_cnn.py --data_dir datasets/videos --epochs 50
```

---

## 🔄 STEP 4: Retrain Gating Classifier (Optional)

If you have videos, retrain the gating classifier with real features:

```bash
python3 main.py train --video_dir datasets/videos --model_out gating_rf_v2.joblib
```

---

## 🎯 STEP 5: Build Ensemble System

Combine all three models:

```bash
# Coming soon - will create inference.py
python3 inference.py --video test_video.mp4
```

This will:
1. Extract features from video
2. Use gating classifier to decide: ViT, CNN, or Both
3. Run the appropriate model(s)
4. Return: REAL or FAKE

---

## 📦 Recommended Datasets

### For Beginners (Start Here):

| Dataset | Size | Download Command |
|---------|------|------------------|
| **Deepfake Images** | 600MB | `kaggle datasets download -d manjilkarki/deepfake-and-real-images -p datasets --unzip` |
| **140k Faces** | 5GB | `kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p datasets --unzip` |

### For Production (More Advanced):

| Dataset | Size | Notes |
|---------|------|-------|
| **DFDC** | 4-470GB | Requires competition acceptance |
| **FaceForensics++** | 2GB+ | Academic benchmark |
| **Celeb-DF** | Variable | Celebrity deepfakes |

---

## 🛠️ Quick Commands Reference

```bash
# Setup Kaggle
./setup_kaggle.sh

# Download dataset
python3 download_datasets.py

# Or manual download (small dataset)
kaggle datasets download -d manjilkarki/deepfake-and-real-images -p datasets --unzip

# Check what you downloaded
ls -lh datasets/

# Train gating classifier on videos (when you have them)
python3 main.py train --video_dir datasets/videos

# Predict for a single video
python3 main.py predict --video_path test.mp4 --model_path gating_rf.joblib --verbose

# Train ViT (coming soon)
# python3 train_vit.py --data_dir datasets/images

# Full inference (coming soon)
# python3 inference.py --video test_video.mp4
```

---

## 📂 Expected Directory Structure

After downloading and organizing:

```
DEEPFAKE_NUS_AI_ML/
├── datasets/
│   ├── images/
│   │   ├── real/
│   │   │   ├── img001.jpg
│   │   │   └── ...
│   │   └── fake/
│   │       ├── img001.jpg
│   │       └── ...
│   └── videos/
│       ├── real/
│       │   ├── video001.mp4
│       │   └── ...
│       └── fake/
│           ├── video001.mp4
│           └── ...
├── models/
│   ├── gating_rf.joblib (✅ Already trained)
│   ├── vit_model.pth (❌ Need to train)
│   └── cnn_model.pth (❌ Need to train)
└── scripts/
    ├── feature_extraction.py (✅ Done)
    ├── gating_model.py (✅ Done)
    ├── main.py (✅ Done)
    ├── train_vit.py (⏳ Coming next)
    └── train_cnn.py (⏳ Coming next)
```

---

## ❓ Troubleshooting

### "401 Unauthorized" Error
```bash
# Check your kaggle.json
cat ~/.kaggle/kaggle.json
# Should show: {"username":"...","key":"..."}

# Re-download token from Kaggle settings
```

### "403 Forbidden" Error for DFDC
```bash
# You need to accept competition rules first
# 1. Go to: https://www.kaggle.com/c/deepfake-detection-challenge
# 2. Click "Join Competition"
# 3. Accept rules
# 4. Then download
```

### "No space left on device"
```bash
# Check available space
df -h

# Use smaller dataset or clean up:
rm -rf datasets/old_data
```

---

## 🎓 What We Trained So Far

### Gating Classifier ✅
- **Purpose**: Routes videos to the right model
- **Input**: 7 video features (resolution, motion, compression, etc.)
- **Output**: "ViT", "CNN", or "ViT + CNN"
- **Performance**: 98% accuracy on synthetic data
- **Status**: ✅ Trained with `demo_train.py`

### ViT Model ❌
- **Purpose**: Detect deepfakes in high-res, static images
- **Input**: Images (224x224 or larger)
- **Output**: Probability of being fake
- **Status**: ❌ NOT TRAINED YET - Need dataset first

### CNN Model ❌
- **Purpose**: Detect deepfakes using temporal patterns
- **Input**: Video frames sequences
- **Output**: Probability of being fake
- **Status**: ❌ NOT TRAINED YET - Need dataset first

---

## 🚀 Next Steps (In Order)

1. **✅ NOW**: Set up Kaggle API
   ```bash
   ./setup_kaggle.sh
   ```

2. **✅ NEXT**: Download a dataset
   ```bash
   python3 download_datasets.py
   # Choose option 3 (600MB dataset) for quick start
   ```

3. **⏳ THEN**: I'll create the ViT training script
   - Will train on downloaded images
   - Takes ~1-2 hours with GPU

4. **⏳ AFTER**: I'll create the CNN training script
   - Will train on video frames
   - Takes ~2-4 hours with GPU

5. **🎯 FINALLY**: Build complete ensemble system
   - Integrate all 3 models
   - Create web UI for testing
   - Deploy!

---

## 💡 Pro Tips

1. **Start Small**: Use the 600MB dataset first to validate everything works
2. **GPU Recommended**: Training ViT/CNN needs GPU (can use Google Colab if needed)
3. **Disk Space**: Large datasets need 10-500GB - check before downloading
4. **Training Time**: 
   - Gating classifier: ~1 minute (CPU) ✅ DONE
   - ViT model: ~1-2 hours (GPU)
   - CNN model: ~2-4 hours (GPU)

---

Ready to proceed? Run:
```bash
./setup_kaggle.sh
```

Then we'll download datasets and train the actual deepfake detection models! 🚀


************************************************************************************
# Hybrid Deepfake Detection: A Gating-Based Ensemble Approach
**Course**: AI/ML Mini Project  
**Date**: December 2025  
**Group Members**: [Member 1], [Member 2], [Member 3]

---

## **Abstract**
The proliferation of hyper-realistic synthesized media, known as "Deepfakes," poses a significant threat to information integrity in the entertainment and media sectors. This project presents a novel hybrid detection system that combines spatial analysis using Vision Transformers (ViT) with temporal analysis using a CNN-LSTM network. A dynamic Gating Mechanism is employed to route input samples to the most appropriate expert model based on feature characteristics. Our approach aims to achieve high detection accuracy while balancing computational efficiency.

---

## **1. Introduction**

### **1.1 Problem Identification**
In the **Entertainment and Media industry**, Generative Adversarial Networks (GANs) have enabled the creation of fake videos that are indistinguishable from reality to the naked eye. While this technology has creative uses (e.g., de-aging actors), it is increasingly weaponized for:
*   **Disinformation**: Fabricating political speeches or news events.
*   **Non-Consensual Media**: Creating fake compromises images or videos of public figures.
*   **Fraud**: Impersonating executives or celebrities for financial gain.

### **1.2 Methodology Overview**
Traditional forensic methods often rely on specific artifacts (e.g., lack of blinking) that modern GANs can now replicate. Our solution employs a **multi-modal ensemble**:
1.  **Spatial Detection**: Identifying pixel-level artifacts in individual frames.
2.  **Temporal Detection**: Identifying movement inconsistencies across a sequence of frames.
3.  **Adaptive Routing**: Using a machine learning classifier to decide which detection path a video should take.

---

## **2. Methodology & System Architecture**

### **2.1 Spatial Analysis: Vision Transformer (ViT)**
Unlike Convolutional Neural Networks (CNNs) that process local features, Transformers utilize self-attention mechanisms to capture global dependencies within an image. We treat image patches as sequences, allowing the model to focus on subtle boundary artifacts often left by face-swapping algorithms.

**Code Snippet: ViT Architecture**
```python
class ViTDeepfakeDetector(nn.Module):
    """
    Vision Transformer for detecting spatial deepfake artifacts.
    Uses a pre-trained ViT base (patch size 16) and adapts the head for binary classification.
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        # Load pre-trained weights from ImageNet
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=2)
        
    def forward(self, x):
        return self.vit(x)
```

### **2.2 Temporal Analysis: CNN-LSTM**
Deepfakes often exhibit temporal flickering or unnatural transitions between frames. We implement a ResNet-18 to extract spatial features from each frame, which are then passed sequentially into an LSTM (Long Short-Term Memory) network to model time-dependent anomalies.

**Code Snippet: CNN-LSTM Architecture**
```python
class ResNetLSTM(nn.Module):
    """
    Hybrid architecture combining CNN for feature extraction and LSTM for temporal modeling.
    """
    def __init__(self, num_classes=2, hidden_dim=256, num_layers=2):
        super(ResNetLSTM, self).__init__()
        # Feature Extractor: ResNet-18 (stripping the classification layer)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_out_dim = resnet.fc.in_features
        
        # Temporal Modeler: LSTM
        self.lstm = nn.LSTM(input_size=self.cnn_out_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        # Classifier Head
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Process every frame through CNN
        c_in = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(c_in)
        features = features.view(batch_size, seq_len, -1)
        
        # Process sequence through LSTM
        lstm_out, _ = self.lstm(features)
        
        # Classify based on the last hidden state
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out
```

### **2.3 Gating Mechanism (The Router)**
To optimize performance, we extract "meta-features" from the video (e.g., brightness variance, noise levels, compression artifacts) and train a **Random Forest Classifier**. This classifier predicts which model (ViT, CNN, or Ensemble) is historically more accurate for that specific type of video.

**Code Snippet: Gating Logic**
```python
def analyze_video(self, video_path):
    # 1. Extract statistical features from the video
    features = extract_video_features(video_path)
    
    # 2. Predict the best expert model to use
    route_pred = self.gating_clf.predict([features])[0]
    route_class = {0: "ViT", 1: "CNN", 2: "Ensemble"}.get(route_pred)
    
    # 3. Route execution
    if route_class == "ViT":
        return self.predict_vit(video_path)
    elif route_class == "CNN":
        return self.predict_cnn(video_path)
    else:
        # Ensemble: Average both predictions
        pred_vit = self.predict_vit(video_path)
        pred_cnn = self.predict_cnn(video_path)
        return (pred_vit + pred_cnn) / 2
```

---

## **3. Implementation Details**

### **3.1 Data Collection & Preprocessing**
We utilize the **Deepfake Detection Challenge (DFDC)** and **FaceForensics++** datasets.
*   **Preprocessing Pipeline**:
    1.  **Frame Extraction**: Sample 1 frame every second.
    2.  **Face Detection**: Use Haar Cascades to crop faces.
    3.  **Normalization**: Resize to 224x224 and normalize using ImageNet statistics `(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
```

---

## **4. Expected Results & Impact**

### **predicted Outcomes**
*   **Accuracy**: We target an AUC score of **0.85+** on the test set.
*   **Robustness**: The ViT component is expected to outperform CNNs on high-resolution fakes, while the LSTM component should tackle low-quality, temporal-glitch fakes.

### **Real-World Application**
This system is designed to be deployed as a **REST API** (using FastAPI) integrated into a web dashboard. Media organizations can upload suspicious footage to the dashboard, where the system provides a probability score and a heatmap of manipulated regions, aiding in the rapid verification of viral content.

---

## **5. Conclusion**
This project demonstrates that a "one-size-fits-all" model is insufficient for the evolving landscape of deepfakes. By combining the local feature extraction of CNNs, the temporal modeling of LSTMs, and the global attention of Transformers—governed by an intelligent gating system—we provide a robust defense against AI-generated misinformation.
