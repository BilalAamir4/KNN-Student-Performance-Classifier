#  KNN Student Performance Classifier

This project implements the **K-Nearest Neighbors (KNN)** algorithm from scratch to classify student performance into categories based on study-related features.

---

##  Project Overview

The goal of this project is to predict a student's performance level:

- 🟢 Good  
- 🟡 Average  
- 🔴 Poor  

based on the following features:

- Hours Studied  
- Sleep Hours  
- Attendance Percentage  
- Previous Scores  

---

## 🧠 Algorithm Used

### K-Nearest Neighbors (KNN)

KNN is a distance-based classification algorithm that:

1. Stores all training data  
2. Calculates distance between a new data point and all training points  
3. Selects the K nearest neighbors  
4. Uses majority voting to determine the class  

Distance is calculated using Euclidean Distance:

d = sqrt(sum((x1 - x2)^2))

---


##  Dataset

The dataset contains the following columns:

- student_id  
- hours_studied  
- sleep_hours  
- attendance_percent  
- previous_scores  
- exam_score  

### 🎯 Target Creation

The target variable is derived from exam_score:

- Good → score ≥ 80  
- Average → 50 ≤ score < 80  
- Poor → score < 50  

---

## ⚙️ Features of This Project

- KNN implemented from scratch (no ML libraries for core logic)  
- Feature scaling using standardization  
- Multi-class classification  
- Model saving using pickle  
- Confusion matrix visualization  

---

## 📈 Results

The model is evaluated using:

- Accuracy Score  
- Confusion Matrix  

All visual outputs are saved in the results/ directory.

---

## ▶️ How to Run

1. Clone the repository

git clone https://github.com/your-username/KNN-Student-Performance-Classifier.git  
cd KNN-Student-Performance-Classifier  

2. Install dependencies

pip install -r requirements.txt  

3. Train the model

cd src  
python train.py  

4. Make predictions

python predict.py  

---

##  Example Prediction

Input:

[5, 7, 80, 75]

Output:

Prediction: Average  

---

## ⚠️ Important Notes

- Feature scaling is critical for KNN performance  
- Choosing the right value of K affects accuracy  
- Larger K gives smoother decision boundaries  
- Smaller K is more sensitive to noise  

---


## 🧑‍💻 Author

Bilal Aamir
