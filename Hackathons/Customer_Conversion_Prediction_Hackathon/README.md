# 🛒 E-Commerce Customer Conversion Prediction

> **Summer Analytics 2026 Mini Hackathon – Week 2**
> Organized by **Consulting & Analytics Club, IIT Guwahati** on **HackerEarth**

## 📌 Overview

This project was developed as part of the **Summer Analytics 2026 Mini Hackathon – Week 2** hosted by the Consulting & Analytics Club, IIT Guwahati on HackerEarth.

The objective was to build a Machine Learning model capable of predicting whether an e-commerce user would **convert into a customer** based on demographic information, browsing behavior, and historical interaction data.

This was my **first Machine Learning hackathon**, and I participated after **just two weeks of learning Machine Learning**, making it an incredible hands-on learning experience.

---

# 🎯 Problem Statement

Given anonymized user data such as:

* Age
* Income
* City Tier
* Device Type
* Traffic Source
* Pages Viewed
* Products Viewed
* Previous Purchases
* Time on Site
* Browser Version
* Campaign Information

the goal was to predict whether a customer would **convert (1)** or **not convert (0)**.

The competition was evaluated using the **F1 Score** on a hidden private test dataset.

---

# 📊 Machine Learning Workflow

The project followed a complete end-to-end Machine Learning pipeline.

### 1. Exploratory Data Analysis (EDA)

* Data inspection
* Missing value analysis
* Distribution analysis
* Feature understanding

---

### 2. Data Preprocessing

* Missing value handling
* Categorical feature encoding
* Feature Scaling using StandardScaler
* Train-validation split (80:20)
* Stratified sampling

---

### 3. Model Development

Model Used:

* Logistic Regression

Parameters:

* max_iter = 1000
* random_state = 42

---

# 📈 Results

| Metric              |      Score |
| ------------------- | ---------: |
| Training Accuracy   | **72.25%** |
| Validation Accuracy | **73.33%** |

The validation accuracy being slightly higher than the training accuracy indicated that the model generalized well without significant overfitting.

The final predictions were exported into the required submission format and successfully accepted on the Hackathon platform.

---

# 📂 Project Structure

```
E-Commerce-Customer-Conversion-Prediction/
│
├── notebook.ipynb
├── Hackathon_Report.pdf
├── submission.csv
├── README.md
└── dataset/
```

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

# 📚 Key Learnings

This project taught me much more than simply training a Machine Learning model.

Some of the biggest takeaways were:

* Performing Exploratory Data Analysis before model building.
* Understanding the importance of feature preprocessing.
* Working with missing values effectively.
* Building an end-to-end Machine Learning pipeline.
* Preparing competition-ready submissions.

One important lesson was related to missing values.

Initially, I chose to remove a feature that contained nearly **20% missing values**. After completing the project and reviewing my approach, I realized that preserving the feature using techniques such as **SimpleImputer** could have retained useful information for the model.

This experience reinforced an important principle:

> **Machine Learning is an iterative process where every experiment and every mistake becomes an opportunity to learn and improve.**

---

# 🚀 Future Improvements

Some improvements I plan to implement include:

* Better Missing Value Imputation
* Advanced Feature Engineering
* Hyperparameter Tuning
* Ensemble Models
* XGBoost
* LightGBM
* CatBoost
* Cross Validation
* Model Explainability using SHAP

---

# 💡 About This Project

Although this project was completed during the second week of my Machine Learning journey, it gave me practical exposure to solving a real-world classification problem under Hackathon conditions.

This experience strengthened my understanding of the complete Machine Learning workflow and motivated me to continue participating in more competitive ML challenges.

---

## 👨‍💻 Author

**Mohammad Owais Ansari**

* Machine Learning Enthusiast
* Data Science Learner
* Engineering Student

⭐ If you found this project interesting, consider giving it a star!
