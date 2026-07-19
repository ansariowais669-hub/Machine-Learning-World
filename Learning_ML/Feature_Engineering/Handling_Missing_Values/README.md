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

### 📌 2. Numerical_Univariate.ipynb
Learned **Numerical Univariate Imputation** techniques, including:
- Mean Imputation
- Median Imputation
- Using `SimpleImputer` from Scikit-learn
- Building preprocessing pipelines with `ColumnTransformer`
- Comparing the effect of different imputation methods on feature distributions

### 📌 3. KNN_Imputer.ipynb
Learned **Multivariate Missing Value Imputation** using the K-Nearest Neighbors approach, including:
- Implementing `KNNImputer` from Scikit-learn to handle missing values by leveraging information from neighbor instances.
- Tuning the imputer by comparing default configuration (`weights='uniform'`) vs. distance-based variation (`weights='distance'`).
- Constructing a baseline evaluation pipeline using `LogisticRegression` on the Titanic dataset.
- Conducting a comparative performance analysis between multivariate KNN imputation and univariate `SimpleImputer` (Mean) based on final model accuracy.

## 🎯 Learning Outcomes

- Understand different types of missing data handling techniques (Deletion, Univariate, and Multivariate).
- Learn when to use deletion vs. statistical imputation vs. algorithmic (KNN) methods.
- Analyze the impact of missing values on data distributions and downstream machine learning models.
- Apply Scikit-learn preprocessing tools (`SimpleImputer`, `KNNImputer`) for efficient data imputation.
- Build a solid foundation for feature engineering in machine learning.

---

**Repository:** Learning_ML  
**Topic:** Feature Engineering → Handling Missing Values