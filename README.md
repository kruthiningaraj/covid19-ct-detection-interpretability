# 🧠 COVID-19 Detection from CT Scans with Interpretability (LIME)

This repository implements the methodology described in the research paper:

📄 **Interpreting Transfer Learning-Based Convolutional Neural Networks for COVID-19 Detection in CT Scan Images**  
DOI: [10.1007/978-3-031-84059-3_17](https://link.springer.com/chapter/10.1007/978-3-031-84059-3_17)

---

## 📜 Overview
This project focuses on:
- Detecting **COVID-19 from CT scans** using **transfer learning CNNs**.
- Applying **LIME interpretability** for explaining model predictions.
- Preprocessing CT scans by **segmenting lung regions** to avoid bias.

---

## 🧠 Workflow
1. **Dataset**: COVID-19 CT scan dataset with labeled COVID-positive and normal cases.
2. **Preprocessing**:
   - Lung segmentation using image masks.
   - Normalization, resizing (224x224).
3. **Model Training**:
   - Transfer learning using pretrained CNNs (VGG-16, VGG-19, InceptionV3).
4. **Interpretability**:
   - Applying **LIME** to visualize regions influencing predictions.
5. **Evaluation**:
   - Metrics: Accuracy, Sensitivity, Specificity, AUC.

---

## 🚀 Getting Started
### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/covid19-ct-detection-interpretability.git
cd covid19-ct-detection-interpretability
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model
```bash
python src/train.py
```

### 4️⃣ Run LIME Interpretability
```bash
python src/lime_explain.py
```

---

## 📜 License
MIT License © 2025
