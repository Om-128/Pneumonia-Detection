# 🫁 Pneumonia Detection Using Transfer Learning

## Overview
This project detects **Pneumonia** from chest X-ray images, classifying them as:  
- **NORMAL**  
- **PNEUMONIA**  

It uses **transfer learning** with **ResNet50** for high performance and faster training.

---

## Dataset
**Kaggle Chest X-ray Dataset**:  
- **Training**: NORMAL 1341, PNEUMONIA 3875  
- **Validation**: 20% split from training  
- **Test**: 624 images  

> ⚠️ Note: Dataset is imbalanced (more Pneumonia images).

---

## Challenges & Solutions
- **Class Imbalance**: Applied dynamic class weights.  
- **Overfitting**: Used data augmentation, early stopping, and learning rate scheduler.  
- **Model Bias to Pneumonia**: Added shuffling, preprocessing, and batch standardization.

---

## Techniques
- **Transfer Learning**: ResNet50 pretrained on ImageNet (base frozen).  
- **Data Augmentation**: Random flips, rotations, zoom.  
- **Preprocessing**: `preprocess_input` + batch dimension.  
- **Pipeline**: Data ingestion → transformation → model training → prediction.  
- **Performance Optimizations**: Shuffling, prefetching, dynamic class weights, early stopping.

---

## Results
- **Test Accuracy**: ~87%  
- **Test Loss**: ~0.31  
- The model predicts both NORMAL and PNEUMONIA reliably.

---

## Tech Stack
- **Programming**: Python  
- **Libraries**: TensorFlow, Keras, Pandas, NumPy, scikit-learn, Pillow  
- **Deployment**: Streamlit  

---

## How to Run
1. Clone the repository:  
   ```bash
   git clone <repo_url>
   cd Pneumonia-Detection
