# Pneumonia Detection Using Transfer Learning

## Project Overview
This project implements a deep learning pipeline to detect **Pneumonia** from chest X-ray images. The goal is to classify X-ray images into two categories:

- **NORMAL**
- **PNEUMONIA**

We leverage **transfer learning** with **ResNet50** as the backbone model to improve performance and reduce training time.

---

## Dataset
The dataset is sourced from **Kaggle Chest X-ray Dataset**:

- **Training Set**:
  - NORMAL: 1341 images
  - PNEUMONIA: 3875 images
- **Validation Set**: 20% split from training set
- **Test Set**: 624 images  

The dataset contains a class imbalance, with Pneumonia images being more than twice the number of Normal images.

---

## Challenges
During development, we faced several challenges:

1. **Class Imbalance**:
   - The dataset has more Pneumonia images than Normal images, which led the model to predict Pneumonia more often.
2. **Overfitting**:
   - Initially, the model achieved high training accuracy but low validation accuracy due to overfitting.
3. **Model not predicting NORMAL**:
   - Early attempts without proper preprocessing, class weights, and shuffling caused the model to predict Pneumonia for almost every test image.

---

## Techniques and Solutions
To address these challenges, we implemented the following techniques:

1. **Transfer Learning**:
   - Used **ResNet50** pretrained on **ImageNet**.
   - The base model weights were initially frozen and fine-tuned later if required.
   
2. **Data Augmentation**:
   - Added random flips, rotations, and zoom to reduce overfitting and improve generalization.
   
3. **Class Weights**:
   - Computed class weights to handle imbalance:
     ```python
     {0: 1.91, 1: 0.68}  # NORMAL, PNEUMONIA
     ```
   - Applied these weights during training to improve prediction for the minority class.

4. **Preprocessing**:
   - Applied `preprocess_input` from `tf.keras.applications.resnet50` to standardize images.
   - Added batch dimension for model input.

5. **Shuffling and Prefetching**:
   - Shuffled the training dataset to avoid model bias from ordered batches.
   - Prefetched batches for faster training.

6. **Early Stopping & Learning Rate Scheduler**:
   - Stopped training when validation loss stopped improving.
   - Reduced learning rate when the model plateaued.

7. **Pipeline Structure**:
   - **Data Ingestion**: Load images and split into training, validation, and test sets.
   - **Data Transformation**: Preprocess images and save a reusable preprocessor.
   - **Model Training**: Build ResNet50-based model and train with class weights.
   - **Prediction Pipeline**: Load trained model and preprocessor for inference on new images.

---

## Results
After applying these techniques:

- **Test Accuracy**: ~87%
- **Test Loss**: ~0.31
- The model can now predict both NORMAL and PNEUMONIA correctly.

---

## How to Run
1. Clone the repository.
2. Install dependencies (TensorFlow, NumPy, scikit-learn, etc.).
3. Run the training pipeline:
   ```bash
   python -m src.pipeline.train_pipeline
