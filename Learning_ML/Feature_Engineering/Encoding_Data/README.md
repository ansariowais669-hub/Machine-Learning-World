# 🏷️ Encoding Data

This folder contains my notes and Jupyter notebooks on **Encoding Data**, an essential part of the **Feature Engineering** pipeline in Machine Learning.

Real-world datasets often contain both **categorical** and **numerical** features that require preprocessing before they can be effectively used by Machine Learning algorithms. In this folder, I explore different encoding and transformation techniques through hands-on implementations using Python and Scikit-learn.

---

# 📘 Notebooks

## 📓 Encoding_Categorical_Data.ipynb

In this notebook, I learned and implemented various techniques for encoding categorical features, including:

- 🔹 Label Encoding
- 🔹 Ordinal Encoding
- 🔹 One-Hot Encoding using Pandas
- 🔹 K-1 (Dummy Variable) One-Hot Encoding
- 🔹 One-Hot Encoding using Scikit-learn
- 🔹 One-Hot Encoding with Top Categories
- 🔹 Using `ColumnTransformer` for preprocessing pipelines

The notebook contains both the theoretical concepts and practical implementations of these encoding techniques on sample datasets.

---

## 📓 Encoding_Numerical_Data.ipynb

In this notebook, I learned techniques used to encode and transform numerical features before model training, including:

### 🔹 Discretization (Binning)

I explored how continuous numerical features can be converted into discrete intervals (bins) using **Scikit-learn's `KBinsDiscretizer`**.

Topics covered include:

- Uniform Binning
- Quantile Binning
- K-Means Binning
- Ordinal Encoding of Bins
- Choosing the number of bins
- Performance comparison before and after discretization
- Using `ColumnTransformer` with `KBinsDiscretizer`

### 🔹 Binarization

I also learned how to convert numerical features into binary values using **Scikit-learn's `Binarizer`**.

Topics covered include:

- Threshold-based feature transformation
- Creating binary features
- Applying `Binarizer` using `ColumnTransformer`
- Comparing model performance before and after binarization

The notebook demonstrates these techniques using practical datasets and evaluates their impact on Machine Learning model performance.

---

# 🎯 Learning Outcomes

After completing these notebooks, I learned:

### Categorical Encoding

- When categorical encoding is required
- The difference between nominal and ordinal categorical variables
- When to use Label Encoding vs Ordinal Encoding
- How One-Hot Encoding works
- The Dummy Variable Trap and K-1 Encoding
- Implementing encoding using **Pandas** and **Scikit-learn**
- Building preprocessing pipelines using `ColumnTransformer`

### Numerical Encoding

- Why numerical features sometimes need transformation
- What discretization (binning) is and when to use it
- Different binning strategies:
  - Uniform
  - Quantile
  - K-Means
- How `KBinsDiscretizer` works
- What binarization is and its practical use cases
- Using `Binarizer` for threshold-based transformations
- Evaluating model performance before and after numerical encoding
- Applying preprocessing using `ColumnTransformer`

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# 📂 Folder Structure

```text
Encoding_Data/
│
├── Encoding_Categorical_Data.ipynb
├── Encoding_Numerical_Data.ipynb
└── README.md
```

---

# 🚀 Topics Covered

### Encoding Categorical Features

- Label Encoding
- Ordinal Encoding
- One-Hot Encoding
- Dummy Variable Encoding (K-1 Encoding)
- One-Hot Encoding using Pandas
- One-Hot Encoding using Scikit-learn
- One-Hot Encoding with Top Categories
- ColumnTransformer

### Encoding Numerical Features

- Discretization (Binning)
- Uniform Binning
- Quantile Binning
- K-Means Binning
- KBinsDiscretizer
- Binarization
- Binarizer
- ColumnTransformer

---

This folder is part of my **Machine Learning** learning repository, where I document my understanding of **Feature Engineering** concepts through practical implementations, experiments, and detailed notes. As I continue learning, this folder will grow with additional encoding techniques and preprocessing methods used in real-world Machine Learning pipelines.