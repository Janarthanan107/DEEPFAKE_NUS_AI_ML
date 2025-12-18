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
