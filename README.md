#Fake-news-prediction-ML-model

# 📰 Fake News Prediction using Machine Learning

This project is a **Fake News Detection system** built using **Machine Learning and Natural Language Processing (NLP)**.  
The model classifies news articles as **REAL (0)** or **FAKE (1)** based on textual content.

---

## 🚀 Project Overview

With the increasing spread of misinformation, fake news detection has become an important real-world problem.  
This project implements an end-to-end NLP pipeline to analyze news data and predict its authenticity.

---

## 📂 Dataset Information

The dataset contains the following columns:

- **id** – Unique ID for a news article  
- **title** – Title of the news article  
- **author** – Author of the article  
- **text** – Main content of the article  
- **label** –  
  - `0` → Real News  
  - `1` → Fake News  

---

## 🛠️ Tech Stack Used

- **Python**
- **NumPy**
- **Pandas**
- **NLTK**
- **Scikit-learn**

---

## 🧠 Approach Used

### 1️⃣ Data Preprocessing
- Handling missing values
- Merging author and title
- Removing special characters using regex
- Converting text to lowercase
- Removing stopwords
- Applying **Porter Stemming**

### 2️⃣ Feature Extraction
- **TF-IDF Vectorization** to convert text into numerical form

### 3️⃣ Model Training
- **Logistic Regression**
- Train-test split: **80% training, 20% testing**

### 4️⃣ Model Evaluation
- Accuracy score used as evaluation metric

---

## 📊 Results

- **Training Accuracy:** ~98.6%
- **Testing Accuracy:** ~97.9%

The model performs well and generalizes effectively on unseen data.

---

## 🔍 Prediction Output

The trained model can predict whether a given news article is **REAL or FAKE** based on its textual content.

---

## ▶️ How to Run the Project

### Step 1: Clone the repository
```bash
git clone https://github.com/your-username/Fake-News-Prediction-ML.git
