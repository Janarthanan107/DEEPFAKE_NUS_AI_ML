# Deepfake Detection System - Project Presentation Report

## Group Details
*(Fill in your group members' names and Student IDs here)*

---

## 1. High-Level Approach

### **Problem Identification**
*   **Industry**: **Entertainment & Media** (and Cybersecurity)
*   **Problem**: The proliferation of hyper-realistic "Deepfake" videos created using Generative Adversarial Networks (GANs) has made it increasingly difficult to distinguish real content from manipulated media.
*   **Significance**:
    *   **Trust in Media**: Deepfakes undermine public trust in news and media (e.g., fake political speeches).
    *   **Identity Theft**: Used for blackmail, fraud, and non-consensual synthetic media.
    *   **Legal & Ethical Risks**: Creators and platforms face liability for hosting harmful manipulated content.
    *   **Challenge**: Traditional digital forensics methods struggle against modern AI-generated fakes which improve rapidly.

### **Methodology and Techniques**
We employ a **hybrid ensemble approach** utilizing state-of-the-art Deep Learning models to capture both spatial (image) and temporal (time-based) anomalies.

*   **Deep Learning Models**:
    1.  **Vision Transformer (ViT)**:
        *   **Purpose**: Analyzes individual video frames to detect varying pixel-level artifacts and spatial inconsistencies that CNNs might miss.
        *   **Architecture**: `vit_base_patch16_224` (or Tiny variant), leveraging self-attention mechanisms to focus on forged regions.
    2.  **CNN-LSTM (ResNet + LSTM)**:
        *   **Purpose**: Analyzes the *sequence* of frames to detect temporal anomalies (e.g., unnatural blinking patterns, jittery lip movements, impossible physics).
        *   **Architecture**: ResNet-18 for feature extraction fed into an LSTM (Long Short-Term Memory) network.
    3.  **Gating Mechanism (Random Forest)**:
        *   **Purpose**: An intelligent router that decides which expert model (ViT or CNN-LSTM) to trust based on the video's input features (e.g., resolution, noise, brightness).

*   **Libraries/Tools**:
    *   **PyTorch**: Core deep learning framework.
    *   **Timm (Hugging Face)**: For pre-trained Vision Transformer models.
    *   **OpenCV**: For video processing and frame extraction.
    *   **Scikit-Learn**: For the Random Forest gating and evaluation metrics.
    *   **FastAPI & React**: For the demonstration prototype.

### **Data Collection**
*   **Data Sources**:
    *   **Kaggle Datasets**: Specifically sourced from the "Deepfake Detection Challenge" and "OpenForensics" datasets.
    *   **FaceForensics++**: A benchmark dataset containing manipulated videos (Deepfakes, Face2Face, FaceSwap).
*   **Data Type**:
    *   **Input**: MP4/AVI Video files.
    *   **Preprocessing**:
        *   Frame extraction (sampling 5-10 frames per video).
        *   Face cropping (using Haar Cascades or detectors).
        *   Normalization and resizing to 224x224 pixels.

---

## 2. Expected Results & Impact

### **Predicted Outcomes**
*   **Goal**: Achieve a detection accuracy of **>85%** on the validation set.
*   **Metrics**:
    *   **Accuracy**: Overall correctness of the model.
    *   **Precision**: Ensuring real videos aren't falsely flagged as fake (False Positives).
    *   **Recall**: Ensuring fake videos aren't missed (False Negatives).
    *   **Routing Efficiency**: Assessing if the Gating model improves performance over a single model baseline.

### **Impact and Application**
*   **Media Integrity**: A tool for social media platforms to automatically flag potential deepfakes before they go viral.
*   **Content Verification Tools**: Can be used by journalists to verify source footage.
*   **Legal Evidence**: Assisting forensic analysts in authenticating video evidence in court.
*   **Scalability**: The hybrid approach allows balancing speed (CNN) vs. accuracy (ViT) depending on the use case.

---

## 3. Presentation Slides Outline

### **Slide 1: Title & Introduction**
*   **Title**: "Unmasking the Fake: A Hybrid Deepfake Detection System"
*   **Context**: The rise of AI-generated media in Entertainment and News.

### **Slide 2: The Problem**
*   **Visual**: A real vs. deepfake comparison (showing how hard it is to tell apart).
*   **Key Issue**: Loss of trust, misinformation, and identity theft.
*   **Why it matters**: The need for automated, high-accuracy detection tools.

### **Slide 3: Challenges**
*   **Technical Challenges**:
    *   Compression artifacts in social media videos hide deepfake traces.
    *   Temporal consistency (smooth movements) in newer GANs.
    *   High computational cost of analyzing every frame.

### **Slide 4: Proposed Solution (Architecture)**
*   **Diagram**: Input Video -> Frame Extraction -> Gating Model -> (Route A: ViT / Route B: CNN-LSTM) -> Final Verdict.
*   **Innovation**: Using a "Gating Model" to dynamically choose the best detector for the specific video type.

### **Slide 5: Demonstration & Results**
*   **Demo**: Show the React Web UI uploading a video and getting a "FAKE" probability.
*   **Results**: Show confusion matrix or accuracy chart.

---

## 4. Jupyter Notebook Demo Checklist
*(To be shown during the live presentation)*

1.  **Setup**:
    *   Install libraries: `torch`, `timm`, `opencv-python`.
    *   Load pre-trained weights (`vit_deepfake.pth`, `cnn_lstm_deepfake.pth`).
2.  **Step 1: Data Loading**:
    *   Load a sample video file (e.g., `test_video.mp4`).
    *   Display a few frames using `matplotlib`.
3.  **Step 2: Preprocessing**:
    *   Show how frames are resized and normalized.
4.  **Step 3: Inference**:
    *   Run the **ViT model** on a single frame.
    *   Run the **CNN-LSTM model** on a sequence of frames.
5.  **Step 4: Result Visualization**:
    *   Print the confidence score: "Probability of Fake: 98.4%".
    *   Overlay the result on the video frame.

---

## Submission Checklist
- [ ] **Presentation Slides (PDF/PPT)**: Upload to Canvas.
- [ ] **Code/Notebooks**: Zip the `frontend`, `api.py`, `inference.py`, and notebooks.
- [ ] **Deadline**: Before **19 Dec / Jan 7.00am**.
