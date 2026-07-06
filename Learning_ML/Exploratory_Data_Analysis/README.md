# 📊 Exploratory Data Analysis (EDA)

Welcome to my **Exploratory Data Analysis (EDA)** learning journey!

This folder documents everything I learn about **EDA**, one of the most important stages in every Machine Learning workflow. Before building models, it is essential to understand the dataset, identify potential issues, discover patterns, and generate insights that guide feature engineering and model selection.

The goal of this folder is not just to write code, but to understand **why each EDA technique is used**, when to apply it, and how it helps in solving real-world Machine Learning problems.

As I continue learning, I will keep expanding this folder with notebooks, visualizations, notes, and practical examples on different datasets.

---

# 📚 Topics I Plan to Cover

- Understanding a Dataset
- Dataset Dimensions
- Data Types
- Missing Values
- Duplicate Values
- Descriptive Statistics
- Correlation Analysis
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
| **Univariate.ipynb** | Learning how to analyze individual categorical and numerical variables using statistical summaries and visualizations. |

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

# 🎯 What I Learned from Lesson 1

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

# 📈 Lesson 2 — Univariate Analysis

After understanding the structure and quality of a dataset, the next step in Exploratory Data Analysis is to study **one feature at a time**.

Univariate Analysis focuses on analyzing the distribution and characteristics of a single variable independently. It helps identify patterns, understand data distribution, detect outliers, measure central tendency, and uncover potential data quality issues before exploring relationships between multiple variables.

In this notebook, I learned how to perform Univariate Analysis on both **categorical** and **numerical** variables using statistical summaries and visualizations.

---

# 📚 Topics Covered

## Categorical Variable Analysis

- Frequency Count
- Percentage Distribution
- Count Plot
- Bar Plot
- Pie Chart

---

## Numerical Variable Analysis

- Histogram
- Distribution Plot (KDE)
- Box Plot
- Mean
- Median
- Mode
- Variance
- Standard Deviation
- Minimum & Maximum Values
- Quartiles
- Interquartile Range (IQR)
- Skewness
- Outlier Detection

---

# 📝 Categorical Variables

Categorical variables contain discrete groups or categories such as Gender, City, Department, or Survival.

Understanding how observations are distributed across categories helps identify class imbalance and dominant categories.

---

## 1. Frequency Count

Count how many observations belong to each category.

```python
df['Column'].value_counts()
```

---

## 2. Percentage Distribution

View category proportions instead of raw counts.

```python
(df['Column'].value_counts() / len(df)) * 100
```

---

## 3. Count Plot

Visualize category frequencies.

```python
sns.countplot(x='Column', data=df)
```

---

## 4. Bar Plot

Another useful visualization for categorical data.

```python
df['Column'].value_counts().plot(kind='bar')
```

---

## 5. Pie Chart

Display the proportion of each category.

```python
df['Column'].value_counts().plot(kind='pie', autopct='%1.1f%%')
```

---

# 📝 Numerical Variables

Numerical variables contain measurable quantities such as Age, Salary, Height, Fare, or Income.

Univariate analysis of numerical features helps understand the spread, center, and shape of the data.

---

## Histogram

Visualize how values are distributed.

```python
sns.histplot(df['Column'])
```

---

## Distribution Plot

Display both the histogram and density curve.

```python
sns.histplot(df['Column'], kde=True)
```

---

## Box Plot

Identify the spread of data and detect outliers.

```python
sns.boxplot(x=df['Column'])
```

---

## Descriptive Statistics

Generate a statistical summary.

```python
df['Column'].describe()
```

Common statistics include:

- Mean
- Median
- Mode
- Minimum
- Maximum
- Quartiles
- Standard Deviation
- Variance

---

# 📸 Notebook Preview

## Categorical Variable Analysis

*(Add screenshots of your notebook here.)*

```markdown
![Categorical Analysis - Part 1](./your_screenshot_name.png)
```

---

## Numerical Variable Analysis

*(Add screenshots of histograms, box plots, and descriptive statistics here.)*

```markdown
![Numerical Analysis - Part 2](./your_screenshot_name.png)
```

---

## Distribution & Outlier Analysis

*(Add screenshots showing distribution plots and box plots.)*

```markdown
![Distribution Analysis - Part 3](./your_screenshot_name.png)
```

---

# 🎯 What I Learned from Lesson 2

From this notebook, I learned how to analyze individual variables before studying relationships between multiple features.

Some important takeaways include:

- Analyze categorical variables using frequency counts and percentages.
- Visualize categorical distributions using count plots, bar plots, and pie charts.
- Study numerical distributions using histograms and KDE plots.
- Use box plots to detect outliers and understand data spread.
- Understand measures of central tendency such as mean, median, and mode.
- Measure variability using variance and standard deviation.
- Identify skewed distributions and unusual observations.
- Build intuition about a dataset before moving to bivariate or multivariate analysis.

These techniques provide a strong foundation for understanding individual features and are essential before performing feature engineering or building Machine Learning models.

---

# 🚀 Learning Goal

My aim is to build a strong intuition for data before building models. I believe that successful Machine Learning starts with a deep understanding of the data, and this repository is my effort to document that learning process from the fundamentals to more advanced Exploratory Data Analysis techniques.

As I continue learning, this folder will grow with additional notebooks covering:

- Bivariate Analysis
- Multivariate Analysis
- Feature Engineering
- Missing Value Treatment
- Outlier Handling
- Data Cleaning
- Statistical Analysis
- Real-world EDA Case Studies

Each notebook represents another step in my journey toward becoming a better Data Scientist and Machine Learning Engineer by learning not just **how** to perform EDA, but also **why** each technique is important.