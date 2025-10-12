# 🫁 Pneumonia Detection Using Transfer Learning

### 🔗 Live Demo  
👉 [Try the App Here](https://pneumonia-prediction-app-om.streamlit.app/)

## 🎥 Demo Video
https://github.com/user-attachments/assets/6346952a-bc2c-433d-ba3b-a401fa173bc0


---

## 📘 Overview
This project uses **Deep Learning** and **Transfer Learning (ResNet50)** to detect **Pneumonia** from chest X-ray images.  
It classifies X-rays into:
- **NORMAL**
- **PNEUMONIA**

Built with a modular ML pipeline for data ingestion, transformation, model training, and prediction.

---

## 📂 Dataset
Sourced from the **Kaggle Chest X-ray Dataset**:
- **Training Set**: 1341 NORMAL, 3875 PNEUMONIA  
- **Test Set**: 624 images  
- **20% validation split** used during training  
- Dataset is **imbalanced**, with more Pneumonia images.

---

## ⚙️ Key Features & Techniques
- ✅ **Transfer Learning** — ResNet50 pretrained on ImageNet  
- 🧩 **Data Augmentation** — random flips, rotations, and zoom  
- ⚖️ **Dynamic Class Weights** — handle class imbalance automatically  
- 🧠 **Preprocessing** — used `preprocess_input` from ResNet50  
- 🚀 **Early Stopping & LR Scheduler** — prevent overfitting  
- 🔄 **Shuffling & Prefetching** — improved data pipeline performance  
- 🧱 **Modular Pipeline** — clean structure for scalability

---

## 📊 Results
| Metric | Value |
|--------|--------|
| **Test Accuracy** | ~87% |
| **Test Loss** | ~0.31 |
| **Performance** | Model predicts both NORMAL and PNEUMONIA accurately |

---

## 💻 Tech Stack
**Languages & Frameworks**  
- Python, TensorFlow, Keras, NumPy, Pandas, scikit-learn  

**Visualization & Deployment**  
- Matplotlib, Seaborn, Streamlit  

**Model**  
- ResNet50 (Transfer Learning)

---

## 🧪 How to Run Locally
```bash
# Clone the repo
git clone https://github.com/<your-username>/Pneumonia-Detection.git
cd Pneumonia-Detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
