# 🎓 Student Dropout and Academic Success Prediction

## 📌 Project Overview

This project uses **Machine Learning** to predict student academic outcomes based on student background, admission information, financial details, and academic performance.

The system predicts one of three outcomes:

* 🔴 **Dropout**
* 🟡 **Enrolled**
* 🟢 **Graduate**

The project also includes an interactive **Streamlit web application** for data visualization, individual student prediction, and batch prediction.

---

## 🎯 Objectives

The main objectives of this project are:

* Analyze student academic and personal data.
* Identify important factors related to student outcomes.
* Build and compare different machine learning classification models.
* Predict whether a student will Dropout, remain Enrolled, or Graduate.
* Develop an interactive web application for making predictions.
* Support early identification of students who may need additional support.

---

## 📊 Dataset

The project uses the **Predict Students' Dropout and Academic Success** dataset.

### Dataset Information

* **Students:** 4,424
* **Features:** 37
* **Target Classes:** 3

  * Dropout
  * Enrolled
  * Graduate

The dataset contains information such as:

* Student demographic information
* Admission details
* Previous qualification
* Admission grade
* Parents' education and occupation
* Scholarship information
* Tuition fee status
* First and second semester academic performance
* Economic indicators such as unemployment, inflation, and GDP

---

## 🔄 Project Workflow

```text
Data Collection
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Data Preprocessing
       ↓
Feature Analysis
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Best Model Selection
       ↓
Streamlit Deployment
       ↓
Student Prediction
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Logistic Regression
* Random Forest
* Extra-Trees

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit

### Model Saving

* Joblib

---

## 🤖 Machine Learning Models

Three classification models were trained and compared:

| Model               | Accuracy |
| ------------------- | -------: |
| Logistic Regression |     ~72% |
| Extra-Trees         |     ~75% |
| Random Forest       |     ~77% |

Based on the experimental results, **Random Forest** was selected as the final model.

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## ⭐ Important Features

The feature analysis showed that academic performance was highly important for predicting student outcomes.

Some important features include:

* Number of subjects passed
* First semester grades
* Second semester grades
* Number of examinations
* Admission grade
* Previous qualification grade
* Age at enrollment

---

# 💻 Streamlit Application

The project includes an interactive Streamlit application with three main features.

## 📈 1. Dashboard

The dashboard provides an overview of the student dataset.

It includes:

* Total number of students
* Number of dropout students
* Number of graduates
* Overall dropout rate
* Student outcome distribution
* Age distribution
* Academic performance analysis

---

## 👨‍🎓 2. Individual Student Prediction

Users can enter information about an individual student, including:

* Age
* Admission grade
* Previous qualification
* Financial information
* Academic performance

The system predicts:

```text
Dropout
Enrolled
Graduate
```

It also displays the predicted probability for each category.

---

## 📂 3. Batch Prediction

Users can upload a CSV file containing multiple student records.

The application:

1. Reads the uploaded CSV file.
2. Processes the student information.
3. Predicts the outcome for each student.
4. Displays the results in a table.
5. Allows the results to be downloaded as a CSV file.

---

## 📁 Project Structure

```text
Students-Dropout-Academic-Success-Prediction/
│
├── Student_Dropout_App/
│   ├── app.py
│   ├── student_dropout_model.pkl
│   └── ...
│
├── notebooks/
│   └── student_dropout_prediction.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ashrif-aham/Students-Dropout-Academic-Success-Prediction.git
```

### 2. Navigate to the project folder

```bash
cd Students-Dropout-Academic-Success-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install the required libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

Navigate to the application folder:

```bash
cd Student_Dropout_App
```

Then run:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📈 Results

The Random Forest model achieved approximately **77% accuracy** on the test dataset.

The model performed particularly well in identifying **Dropout** and **Graduate** students, while the **Enrolled** category was more difficult to classify.

This shows that early academic performance can provide useful information for predicting student outcomes.

---

## 👨‍💻 Author

**Ashrif Ahamed**

Data Science Undergraduate
Sabaragamuwa University of Sri Lanka

GitHub:
https://github.com/Ashrif-aham

---

## 📜 License

This project is developed for **academic and educational purposes**.
