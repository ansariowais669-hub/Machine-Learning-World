# 📐 Mathematical Transformations

Welcome to the **Mathematical_Transformations** section of my **Feature Engineering** journey!

This folder contains my notes and implementations of **mathematical feature transformation techniques** used to improve the distribution of numerical features, reduce skewness, and enhance the performance of machine learning models.

---

## 📘 Notebook Included

### 🔹 Function_Transformer.ipynb

In this notebook, I explored how to apply mathematical transformations using Scikit-learn's `FunctionTransformer`.

### Topics Covered

- Understanding why mathematical transformations are required
- Handling skewed numerical features
- Applying logarithmic (`log1p`) transformation
- Using `FunctionTransformer` from Scikit-learn
- Applying transformations to selected columns with `ColumnTransformer`
- Comparing feature distributions before and after transformation
- Visualizing distributions using PDF plots and Q-Q plots
- Evaluating the impact of transformations on different machine learning models

---

## 🛠️ Libraries Used

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SciPy

---

## 🎯 Learning Outcome

Through this notebook, I learned how mathematical transformations can make numerical features more suitable for machine learning algorithms by reducing skewness and making their distributions closer to normal. I also understood that these transformations tend to benefit linear models more than tree-based models, which are generally less sensitive to feature distributions.

---

More mathematical transformation techniques will be added to this folder as I continue my Feature Engineering journey.