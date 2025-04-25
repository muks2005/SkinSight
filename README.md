# SkinSight
A deep learning project that uses CNNs to detect and classify skin lesions from dermatoscopic images. Trained on the HAM10000 dataset, it helps identify conditions like melanoma, nevus, and keratosis—aiming to make skin diagnostics smarter and more accessible.
---

# Skin Cancer Classification using CNN

> A deep learning project for classifying skin lesions using the HAM10000 dataset, achieving **76% accuracy** using a custom-built Convolutional Neural Network.

---

## 📌 Project Overview

This project demonstrates the use of a **Convolutional Neural Network (CNN)** to classify images of skin lesions into one of seven diagnostic categories. The goal is to aid early detection of various types of skin cancer through image classification.

- ✅ Model Accuracy: **76%**
- 🧠 Model Type: Custom-built CNN (Keras + TensorFlow)
- 🖼️ Dataset: HAM10000 (10,000+ dermatoscopic images)
- ⏳ Training: 30 Epochs
- 🔬 Classes: 7 Skin Disease Types

---

## 📊 Dataset - HAM10000

The **Human Against Machine with 10000 training images (HAM10000)** dataset is a large, curated collection of multi-source dermatoscopic images of pigmented lesions.

**Key Features:**
- 📁 10,000+ high-resolution images
- 🩺 7 diagnostic categories:
  - Actinic keratoses (akiec)
  - Basal cell carcinoma (bcc)
  - Benign keratosis-like lesions (bkl)
  - Dermatofibroma (df)
  - Melanocytic nevi (nv)
  - Vascular lesions (vasc)
  - Melanoma (mel)
- 🧬 Images include both metadata and labels

---

## 🛠️ Project Workflow

### 1. Data Preprocessing
- Images resized and normalized (0–1 scale)
- Label encoding via one-hot encoding
- Dataset split into training and validation sets

### 2. CNN Architecture
- Multiple **Conv2D → MaxPooling** blocks
- Flatten + Dense layers
- Dropout for regularization
- Compiled with:
  - Optimizer: `Adam`
  - Loss: `categorical_crossentropy`
  - Metrics: `accuracy`

### 3. Model Training
- Trained over **30 epochs** on 10k+ images
- Batch size optimized for performance
- Real-time accuracy and loss tracking via Keras callbacks

### 4. Model Evaluation
- Accuracy on validation/test set: **76%**
- **Confusion Matrix** generated
- **Precision, Recall, and F1-Score** calculated per class
- Visualizations to compare prediction vs. truth

---

## 📈 Results & Insights

| Metric     | Value |
|------------|-------|
| Accuracy   | 76%   |
| Precision  | Varies per class |
| Recall     | Varies per class |
| F1-Score   | Varies per class |

- The model shows **strong performance on common classes** like `nv` and `mel`.
- Less frequent classes (like `df` and `vasc`) could benefit from **data augmentation** or **class balancing**.
- Confusion matrix analysis highlights **areas of confusion** between visually similar lesion types.

---

## 🧪 Future Improvements

- ✅ Implement **data augmentation** (rotation, flipping, zoom)
- 🧠 Try deeper architectures (ResNet, EfficientNet)
- 🔁 Use transfer learning with pre-trained models
- ⚖️ Handle class imbalance with SMOTE or weighted loss
- 🧾 Add AUC/ROC metrics for more robust evaluation

---

## ▶️ Running the Code (Google Colab)

1. Open this notebook in [Google Colab](https://colab.research.google.com/drive/15uefTPr7ztxjgMFj9cKqZYXA9LQfsr5L).
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Install required libraries (if not already present):
   ```bash
   !pip install tensorflow pandas opencv-python matplotlib seaborn scikit-learn
   ```
4. Load the dataset (CSV + Images) from Drive.
5. Run each cell sequentially — the model will be trained and evaluated automatically.

---

##  Folder Structure (Suggested)

```
SkinSight/
│
├── dataset/
│   ├── hmnist_28_28_RGB.csv
│   └── images/
│
├── model/
│   └── skin_cancer_model.h5
│
├── notebooks/
│   └── skin_cancer_classification.ipynb
│
├── README.md
└── requirements.txt
```

---

##  Disclaimer

This model is for **educational and experimental purposes only**. It is **not a substitute** for professional medical diagnosis. Always consult certified medical professionals for health-related issues.


##  Acknowledgements

- [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- TensorFlow & Keras documentation
- Google Colab


