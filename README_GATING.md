# Gating Classifier for Deepfake Detection

This system intelligently routes videos to the appropriate deepfake detection model (ViT, CNN, or both) based on video characteristics.

## 🎯 Overview

The gating classifier analyzes video features and decides:
- **ViT**: Best for static, high-resolution videos
- **CNN**: Best for dynamic, motion-heavy videos
- **ViT + CNN**: Best for mixed characteristics (ensemble approach)

## 📁 File Structure

```
├── feature_extraction.py  # Extract video features (resolution, motion, compression)
├── gating_model.py        # Train/load Random Forest classifier
├── rule_based.py          # Fallback rule-based decision system
├── main.py                # CLI for training and prediction
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Classifier

**Option A: With labeled data (CSV)**
```bash
python main.py train --video_dir /path/to/videos --labels_csv labels.csv --model_out gating_rf.joblib
```

CSV format:
```csv
video_filename,label
video1.mp4,ViT
video2.mp4,CNN
video3.mp4,ViT + CNN
```

**Option B: Without labels (uses rule-based auto-labeling)**
```bash
python main.py train --video_dir /path/to/videos --model_out gating_rf.joblib
```

### 3. Predict for New Videos

**With trained model:**
```bash
python main.py predict --video_path test_video.mp4 --model_path gating_rf.joblib --verbose
```

**Without model (rule-based):**
```bash
python main.py predict --video_path test_video.mp4 --verbose
```

## 🔬 Feature Extraction

The system extracts 7 features from each video:

| Feature | Description |
|---------|-------------|
| `area_log` | Log of average frame resolution |
| `aspect_std` | Standard deviation of aspect ratios |
| `motion` | Optical flow magnitude (motion intensity) |
| `blur_inv` | Inverse blur measure (sharpness) |
| `blockiness` | Compression artifact measure |
| `fps_log` | Log of frames per second |
| `bitrate_log` | Log of video bitrate |

## 🧠 Decision Logic

### Rule-Based System
When no trained model is available:
- **CNN**: `motion > 0.8 AND (blur_inv > 0.8 OR blockiness > 3.0)`
- **ViT**: `area_log > 12.0 AND motion < 0.5 AND blur_inv < 0.5`
- **ViT + CNN**: All other cases

### ML-Based System
Random Forest classifier trained on video features to predict optimal model choice.

## 📊 Example Usage

```python
from feature_extraction import extract_video_features
from gating_model import predict_with_confidence

# Extract features
features = extract_video_features("my_video.mp4")

# Predict
model, confidence = predict_with_confidence(features, "gating_rf.joblib")
print(f"Use: {model}")
print(f"Confidence: {confidence}")
```

## 🎓 Integration with Deepfake Models

Once you have predictions, route videos accordingly:

```python
if model == "ViT":
    result = vit_model.predict(video)
elif model == "CNN":
    result = cnn_model.predict(video)
else:  # "ViT + CNN"
    vit_result = vit_model.predict(video)
    cnn_result = cnn_model.predict(video)
    result = ensemble(vit_result, cnn_result)
```

## 🔧 Customization

### Adjust Feature Extraction
Edit `feature_extraction.py` to add more features (e.g., frequency domain, face detection metrics)

### Change Classifier
Replace Random Forest in `gating_model.py` with:
- XGBoost
- Neural Network
- Mixture-of-Experts (MoE)

### Tune Rules
Modify thresholds in `rule_based.py` based on your dataset characteristics

## 📈 Performance Tips

1. **More training data = better predictions**
2. **Balance classes** (equal examples of ViT, CNN, ViT+CNN)
3. **Feature normalization** may improve RF performance
4. **Cross-validation** to tune hyperparameters

## ❓ Troubleshooting

**Issue**: "Could not extract features"
- Check video file is not corrupted
- Ensure OpenCV can read the video format

**Issue**: "No valid videos found"
- Verify video_dir path is correct
- Check videos are `.mp4` format (or update glob pattern)

**Issue**: Low accuracy
- Collect more labeled training data
- Review rule-based thresholds
- Consider deep learning gating model

## 🚀 Next Steps

1. **Collect labeled data** for your specific dataset
2. **Train the classifier** with real examples
3. **Integrate with your ViT and CNN models**
4. **Evaluate end-to-end performance**
5. **Deploy in production** 🎉

---

Built for NUS AI/ML Deepfake Detection System 🔬
