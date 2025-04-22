# 🩺 Breast Cancer Detection & Segmentation Using Machine Vision

A machine vision project designed to automate the classification and segmentation of breast cancer using ultrasound images with both classical machine learning and deep learning methods.

---

## 📌 Overview

This project assists medical professionals by providing accurate and efficient methods for:
- Classifying breast tumors as benign, malignant, or normal.
- Segmenting and identifying tumor regions using deep learning techniques.

---

## 🗃️ Dataset

- **Source:** [Al-Dhabyani et al., 2020](https://doi.org/10.1016/j.dib.2019.104863)
- **Type:** Ultrasound Images (PNG format, ~500x500 resolution)
- **Categories:** Benign, Malignant, Normal
- **Samples:** 780 images

---

## 🛠️ Technologies & Methods

- **Feature Extraction:** VGG16 (Transfer Learning)
- **Classification Models:**
  - Support Vector Machine (SVM)
  - Decision Tree (DT)
  - Random Forest (RF)
  - AdaBoost
- **Segmentation Models:** U-Net enhanced with MobileViTv2
- **Libraries:** Python, NumPy, Pandas, TensorFlow, Keras, scikit-learn

---

## 🔧 Installation

Clone the repository and install the required packages:
```bash
git clone https://github.com/yourusername/breast-cancer-machine-vision.git
cd breast-cancer-machine-vision
pip install -r requirements.txt
```

---

## 🚦 Usage

1. **Feature Extraction:**
```bash
python feature_extraction.py
```

2. **Training Models:**
```bash
python train_model.py --model svm
```

3. **Segmentation with U-Net:**
```bash
python segment.py --image path/to/image.png
```

---

## 📈 Results

### Classification Performance:

| Model          | Accuracy |
|----------------|----------|
| SVM            | 87.5%    |
| Decision Tree  | 93.75%   |
| Random Forest  | 96.15%   |
| AdaBoost       | 92.5%    |

### Segmentation Performance (U-Net):

| Metric    | Score  |
|-----------|--------|
| Dice      | 0.8166 |
| IoU       | 0.6901 |
| Precision | 0.8256 |
| Recall    | 0.8079 |

---

## 🎯 Future Improvements

- Explore advanced architectures like Swin Transformer or Nested U-Net.
- Enhance model generalizability with more diverse datasets.
- Improve user interfaces for easier clinical integration.

---

## 📚 References & Resources

- [Kaggle Breast Ultrasound Dataset](https://www.kaggle.com/code/omkarmodi/vgg-19-feature-extraction)
- [U-Net Biomedical Segmentation Guide](https://www.kaggle.com/code/saidislombek/biomedical-image-segmentation-with-u-net)



