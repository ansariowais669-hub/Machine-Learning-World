# Handling Missing Values

This folder contains notebooks covering fundamental techniques for handling missing data during feature engineering. The notebooks focus on understanding the impact of missing values and implementing practical imputation strategies commonly used in machine learning workflows.

## 📚 Notebooks

### 📌 1. Complete_Case_Analysis.ipynb
Learned how to perform **Complete Case Analysis (Listwise Deletion)**, including:
- Identifying missing values in datasets
- Selecting features suitable for complete case analysis
- Removing rows with missing values
- Comparing feature distributions before and after deletion
- Understanding when complete case analysis is appropriate and its limitations

---

### 📌 2. Numerical_Univariate.ipynb
Learned **Numerical Univariate Imputation** techniques, including:
- Mean Imputation
- Median Imputation
- Using `SimpleImputer` from Scikit-learn
- Building preprocessing pipelines with `ColumnTransformer`
- Comparing the effect of different imputation methods on feature distributions

---

### 📌 3. KNN_Imputer.ipynb
Learned **Multivariate Missing Value Imputation** using the K-Nearest Neighbors approach, including:
- Implementing `KNNImputer` from Scikit-learn to handle missing values using information from neighboring samples.
- Comparing different weighting strategies (`weights='uniform'` vs. `weights='distance'`).
- Building a machine learning pipeline with `LogisticRegression`.
- Evaluating KNN Imputation against Mean Imputation using model accuracy.

---

### 📌 4. MICE_Iterative_Imputer_Multivariate.ipynb
Learned **MICE (Multiple Imputation by Chained Equations)** and **Iterative Imputation** concepts, including:
- Understanding the intuition behind **Iterative (MICE) Imputation** for multivariate missing data.
- Learning how each feature with missing values can be predicted using the remaining available features.
- Performing iterative updates where missing values are repeatedly refined until convergence.
- Building regression models to estimate missing numerical values.
- Understanding the complete iterative workflow through a manual implementation using `LinearRegression`.
- Comparing iterative multivariate imputation with simpler statistical imputation techniques.

---

## 🎯 Learning Outcomes

- Understand different missing data handling strategies, including **Deletion**, **Univariate Imputation**, and **Multivariate Imputation**.
- Learn when to use Complete Case Analysis and when imputation techniques are more appropriate.
- Apply statistical imputation methods such as Mean and Median Imputation.
- Use advanced multivariate techniques including **KNN Imputation** and **MICE (Iterative Imputation)**.
- Understand the iterative process behind MICE and how regression models can estimate missing values.
- Analyze how different imputation techniques affect feature distributions and machine learning model performance.
- Build preprocessing pipelines using Scikit-learn tools for robust and reproducible machine learning workflows.
- Develop a strong foundation in handling missing values as part of feature engineering.

---

**Repository:** Learning_ML  
**Topic:** Feature Engineering → Handling Missing Values