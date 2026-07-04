# 📊 Exploratory Data Analysis (EDA)

Welcome to my **Exploratory Data Analysis (EDA)** learning journey!

This folder documents everything I learn about **EDA**, one of the most important stages in every Machine Learning workflow. Before building models, it is essential to understand the dataset, identify potential issues, discover patterns, and generate insights that guide feature engineering and model selection.

The goal of this folder is not just to write code, but to understand **why each EDA technique is used**, when to apply it, and how it helps in solving real-world Machine Learning problems.

As I continue learning, I will keep expanding this folder with notebooks, visualizations, notes, and practical examples on different datasets.

---

# 📚 Topics I Plan to Cover

- Understanding a dataset
- Dataset dimensions
- Data types
- Missing values
- Duplicate values
- Descriptive statistics
- Correlation analysis
- Univariate Analysis
- Bivariate Analysis
- Multivariate Analysis
- Data Visualization
- Outlier Detection
- Feature Relationships
- Data Cleaning
- Feature Engineering
- Statistical Analysis
- Real-world EDA Case Studies

---

# 📂 Contents

| Notebook | Description |
|----------|-------------|
| **understanding_data.ipynb** | Learning the fundamental steps of understanding a new dataset before performing any analysis. |

More notebooks will be added as I progress.

---

# 📝 Lesson 1 — Understanding a Dataset

The first notebook focuses on the very first questions every data scientist should ask after loading a dataset.

## Questions to Ask

### 1. How big is the dataset?

Understand the number of rows and columns.

```python
df.shape
```

---

### 2. What does the dataset look like?

View a few random observations.

```python
df.sample(5)
```

---

### 3. What are the data types of each column?

Understand which features are numerical and which are categorical.

```python
df.info()
```

---

### 4. Are there any missing values?

Identify incomplete data before preprocessing.

```python
df.isnull().sum()
```

---

### 5. How does the data look mathematically?

Generate descriptive statistics.

```python
df.describe()
```

---

### 6. Are there duplicate rows?

Check whether duplicate observations exist.

```python
df.duplicated().sum()
```

---

### 7. How are numerical columns related?

Measure correlation between numerical features.

```python
df.corr(numeric_only=True)
```

or

```python
df.corr()['Target_Column']
```

---

# 📸 Notebook Preview

## Understanding the Dataset

![Understanding Data - Part 1](./Screenshot%202026-07-05%20014158.png)

---

## Data Types & Missing Values

![Understanding Data - Part 2](./Screenshot%202026-07-05%20014206.png)

---

## Descriptive Statistics

![Understanding Data - Part 3](./Screenshot%202026-07-05%20014218.png)

---

## Duplicate Values & Correlation

![Understanding Data - Part 4](./Screenshot%202026-07-05%20014227.png)

---

# 🎯 What I Learned

From this lesson, I learned how to quickly inspect any dataset by answering a few important questions before jumping into visualization or model building.

- Determine dataset size using `shape`
- Preview observations using `head()`, `tail()`, or `sample()`
- Inspect column data types with `info()`
- Detect missing values with `isnull().sum()`
- Generate descriptive statistics using `describe()`
- Check duplicate records
- Explore relationships between numerical variables using correlation

These steps form the foundation of Exploratory Data Analysis and should be performed whenever working with a new dataset.

---

# 🚀 Learning Goal

My aim is to build a strong intuition for data before building models. I believe that good Machine Learning starts with good data understanding, and this repository is my effort to document that learning process.

This folder will continue to grow as I explore more EDA techniques, apply them to diverse datasets, and deepen my understanding of data analysis.