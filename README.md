# 🖐️ Sign Language Recognition (A–Z) using MediaPipe & Machine Learning

## 📌 Project Overview
This project is a real-time Sign Language Recognition system that identifies hand gestures for alphabets (A–Z) using hand landmark detection and a machine learning model.

Instead of using raw images, the system extracts hand skeleton (landmarks) using MediaPipe and classifies them using a trained neural network, making it efficient and robust for real-time applications.

---

## 🚀 Features
- Real-time hand gesture recognition (A–Z)
- Uses MediaPipe for accurate hand tracking
- Lightweight ML model (MLPClassifier)
- High accuracy (~95% with cross-validation)
- Works on webcam input
- Fast and efficient (no GPU required)

---

## 🧠 Tech Stack
- Python 3.11
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy
- Pandas

---

## 📂 Project Structure

SLRW/
│── extract_landmarks.py
│── train_model.py
│── validation.py
│── predict_webcam.py
│── model.pkl
│── label_encoder.pkl
│── scaler.pkl
│── train_requirements.txt
│── README.md

---

## ⚙️ How It Works

1. Landmark Extraction  
   - MediaPipe detects 21 hand landmarks  
   - Each landmark has (x, y, z) coordinates  
   - Total features = 63  

2. Preprocessing  
   - Normalization using wrist-relative coordinates  
   - Scaling using StandardScaler  

3. Model Training  
   - MLPClassifier (Neural Network)  
   - Class balancing applied  
   - Evaluated using 5-fold cross-validation  

4. Prediction  
   - Webcam captures hand  
   - Landmarks extracted in real-time  
   - Model predicts alphabet (A–Z)  

---

## 📊 Model Performance
- Cross-validation accuracy: ~95%  
- Stable performance across folds  

---

## ▶️ How to Run

1. Create virtual environment  
python -m venv venv  

2. Activate environment (Windows)  
venv\Scripts\activate  

3. Install dependencies  
pip install -r train_requirements.txt  

4. Run webcam prediction  
python predict_webcam.py  

---

## 🎯 Use Cases
- Assistive technology for communication  
- Human-computer interaction  
- Gesture-based control systems  

---

## 📌 Future Improvements
- Add support for numbers (0–9)  
- Build a web app using Flask  
- Improve accuracy with more diverse data  

