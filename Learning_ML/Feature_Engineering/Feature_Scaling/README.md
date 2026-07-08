# 📏 Feature Scaling

This folder contains my notes and Jupyter notebooks on **Feature Scaling**, one of the most fundamental steps in **Feature Engineering**.

Feature scaling transforms numerical features to a common scale, improving the performance, stability, and convergence of many Machine Learning algorithms. Each notebook combines theoretical concepts with practical implementation using Python and Scikit-learn.

## 📚 Learning Roadmap

### ✅ Feature Scaling - 1 : Standardization (Z-Score Scaling)

In this notebook, I covered:

- Why feature scaling is important
- Introduction to Standardization
- Working of `StandardScaler`
- Fitting the scaler on training data only
- Transforming training and test datasets correctly
- Visualizing data before and after scaling
- Comparing feature distributions
- Effect of Standardization on Logistic Regression
- Understanding the impact of outliers on Standardization

---

### ✅ Feature Scaling - 2 : Min-Max Scaling (Normalization)

In this notebook, I covered:

- Introduction to Min-Max Scaling
- Working of `MinMaxScaler`
- Scaling features to the **[0, 1]** range
- Applying scaling on training and test datasets
- Visualizing feature distributions before and after scaling
- Comparing data distributions before and after normalization
- Understanding how Min-Max Scaling preserves the shape of the original distribution
- Learning when Min-Max Scaling is preferred and its sensitivity to outliers

---

### 🚧 Coming Next

- Max Absolute Scaling
- Robust Scaling
- Practical comparison of all scaling techniques
- Choosing the right scaling method for different datasets and machine learning algorithms

---

## 🎯 Goal

The goal of this section is to develop a strong understanding of feature scaling techniques, learn when each method should be used, and understand their impact on data preprocessing and machine learning model performance through hands-on implementation.