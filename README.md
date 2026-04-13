# 🛢️ AI SpillGuard — Oil Spill Detection using Deep Learning

An AI-powered web application for detecting and segmenting oil spills from aerial and drone imagery using U-Net (PyTorch).

---

## 🚀 Live Demo  
👉 https://aispillguardoilspilldetection-rkmmsxwysznvk3app-navya.streamlit.app/

---

## 📌 Project Overview

Oil spills are a major environmental hazard affecting marine ecosystems. Rapid detection is critical.

This project uses Deep Learning (U-Net) to:
- Detect oil spills
- Segment affected regions
- Provide real-time visualization

---

## 🧠 Model Details

- Architecture: U-Net (PyTorch)
- Input Size: 256 × 256 (BGR)
- Loss Function: BCE + Dice Loss
- Optimizer: Adam (lr = 1e-4)
- Epochs: 25
- Batch Size: 16

---

## 📊 Performance

| Metric | Value |
|------|------|
| Best Validation Dice | 0.9189 |
| Test Dice Score | 0.9058 |
| Test Loss | 0.2705 |

---

## 📁 Project Structure

```
oil-spill-prediction/
│
├── app.py                     # Streamlit App
├── best_model.pth            # Trained Model
├── requirements.txt          # Dependencies
├── OilSpill_Detection.ipynb  # Training Notebook
│
├── oil_spill_dataset/        # Dataset
├── oil_spill_augmented/      # Augmented Data
│
├── eda_augmented.png
├── training_curves.png
├── metrics_chart.png
├── visualization_results.png
├── training_summary.txt
└── oil-spill.zip
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Navya032006/AI_SpillGuard_OilSpillDetection.git
cd AI_SpillGuard_OilSpillDetection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**

* streamlit
* torch
* torchvision
* numpy
* Pillow
* matplotlib
* opencv-python-headless

---

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 🖼️ How It Works

* Upload an aerial/drone image
* Image is converted to **BGR format (OpenCV)**
* Passed through trained U-Net model

**Outputs:**

* Segmentation mask
* Oil coverage %
* Confidence score
* Heatmap visualization

---

## 🔍 Key Features

* ✅ Real-time oil spill detection
* ✅ Segmentation + heatmap visualization
* ✅ Adjustable confidence threshold
* ✅ Clean Streamlit UI
* ✅ Deployable web app
* ✅ GPU-trained model

---

## ⚠️ Model Limitations & Errors

Even with good performance, the model has some real-world limitations:

* **False Positives:** Confuses oil with algae, reflections, or shadows
* **False Negatives:** Misses some colored (green/orange) oil spills
* **Partial Detection:** May not capture full spill shape accurately
---

## 🛠️ Fixes & Improvements

### ✅ Immediate Fix (No retraining)
### ⚠️ Why False Detections Occur

- The model learns patterns only from the training data  
- It has seen oil spills and clean water, but not confusing patterns like:
  - Algae  
  - Sunlight reflections  
  - Ocean glare  

👉 As a result, it may interpret **similar textures or colors as oil**, leading to false positives.

---
Increase threshold:

```python
threshold = 0.65
```

✔️ Reduces false positives
✔️ Improves precision

---

### 🚀 Dataset Improvements (Most Important)

Add 30–50 negative examples:

* Algae
* Coral reefs
* Sunlight glare
* Ocean reflections

(All with black masks — no oil)

👉 This alone can fix ~80% errors

---

### 🎨 Improve Color Generalization

Add Color Jitter Augmentation:

* Hue shift
* Saturation change

👉 Helps detect:

* Green oil
* Orange oil
* Rainbow oil

---

## 📈 Results Visualization

The app provides:

* Original Image
* Prediction Heatmap
* Oil Mask Overlay

---

## 🎯 Future Enhancements

* Reduce false positives
* Train with diverse datasets
* Add SAR satellite images
* Improve boundary accuracy
* Deploy mobile-friendly version

---

## 👩‍💻 Author

Navya Sai
B.Tech CSE (3rd Year)

---

## 🌊 Final Conclusion

The model performs well (Dice ≈ 0.91), and observed errors are:

* Expected
* Explainable
* Fixable

👉 The model learned what oil looks like,
but needs more data to learn what oil is NOT

---
