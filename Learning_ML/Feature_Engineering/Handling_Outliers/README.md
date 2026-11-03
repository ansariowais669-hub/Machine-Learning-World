# Handling Outliers

This folder contains notebooks demonstrating the fundamental concepts, detection techniques, and practical implementation of **outlier detection** in Machine Learning. It explains what outliers are, why they occur, their impact on data analysis, and multiple approaches to detect and handle them using widely adopted statistical methods.

The notebooks cover different statistical approaches for detecting outliers — **Z-Score, Interquartile Range (IQR), and Percentile-Based Detection** — and demonstrate how to treat outliers using **Trimming** and **Capping (Winsorization)** to build more robust machine learning models.

## Topics Covered

- Introduction to Outliers
- Causes and Impact of Outliers
- Common Outlier Detection Techniques
- Detecting Outliers using **Z-Score**
- Calculating Mean and Standard Deviation
- Identifying Outliers using the **3-Sigma Rule**
- Detecting Outliers using **Interquartile Range (IQR)**
- Calculating Quartiles, IQR, and Outlier Bounds
- Detecting Outliers using **Percentile-Based Methods**
- Calculating Upper and Lower Percentile Limits
- Handling Outliers using **Trimming**
- Handling Outliers using **Capping (Winsorization)**
- Comparing Different Outlier Handling Approaches
- Visualizing Data Distributions using Histograms and Boxplots
- Practical Implementation using Python, NumPy, Pandas, Matplotlib, and Seaborn

## Notebooks

| Notebook | Description |
|----------|-------------|
| **OutlierAnalysis.ipynb** | Introduces outliers, explains their causes and impact, and demonstrates outlier detection using both **Z-Score** and **Interquartile Range (IQR)** methods with Python and NumPy. |
| **InterQuartileRange_Trimming&Capping_Outlier.ipynb** | Demonstrates how to handle outliers detected using the **IQR** method through **Trimming** (removing outliers) and **Capping** (replacing extreme values with boundary values). Includes practical implementation and comparison of both techniques. |
| **Z-score_Trimming&Capping_Outlier.ipynb** | Demonstrates how to detect outliers using the **Z-Score (3-Sigma Rule)** and handle them using **Trimming** and **Capping (Winsorization)**. Includes visualization of data distribution before and after treatment, along with practical implementation using Python. |
| **Percentile_Trimming&Winsorization_Outlier.ipynb** | Demonstrates percentile-based outlier handling using **Trimming** and **Winsorization**. Calculates the 1st and 99th percentile limits to identify extreme values in the Height column of a dataset. Implements Trimming by removing observations outside these limits and Winsorization by capping extreme values at the calculated boundaries. Includes distribution plots and boxplots to visualize the effects of outlier treatment. |

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Learning Outcomes

After completing these notebooks, you will be able to:

- Understand what outliers are and why they matter.
- Explain the impact of outliers on statistical analysis and machine learning models.
- Detect outliers using the **Z-Score** method.
- Apply the **3-Sigma Rule** to identify extreme observations.
- Compute quartiles and the **Interquartile Range (IQR)**.
- Determine lower and upper bounds for identifying outliers.
- Understand percentile-based outlier detection.
- Calculate percentile limits to identify extreme observations.
- Handle outliers using **Trimming** and **Capping** techniques.
- Apply **Winsorization** using percentile-based boundaries.
- Compare outlier treatment using **Z-Score, IQR, and Percentile-Based Methods**.
- Understand when to remove outliers versus when to cap them.
- Visualize data distributions before and after outlier treatment using Matplotlib and Seaborn.
- Implement different outlier detection and treatment techniques in Python using real datasets.