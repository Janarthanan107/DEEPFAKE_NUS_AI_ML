# 🚀 Complete Setup Guide - Deepfake Detection System

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
