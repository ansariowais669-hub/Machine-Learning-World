# Handling Outliers

This folder contains notebooks demonstrating the fundamental concepts, detection techniques, and practical implementation of **outlier detection** in Machine Learning. It covers what outliers are, why they occur, their impact on data analysis, and multiple approaches to detect and handle them using widely adopted statistical methods.

In addition to identifying outliers using statistical techniques, this folder also explains how to **treat outliers** using **Trimming** and **Capping**, helping build more robust machine learning models.

## Topics Covered

- Introduction to Outliers
- Causes and Impact of Outliers
- Common Outlier Detection Techniques
- Detecting Outliers using **Z-Score**
- Detecting Outliers using **Interquartile Range (IQR)**
- Calculating Quartiles, IQR, and Outlier Bounds
- Identifying Outliers from a Sample Dataset
- Handling Outliers using **Trimming**
- Handling Outliers using **Capping (Winsorization)**
- Comparing Different Outlier Handling Approaches
- Practical Implementation using Python and NumPy

## Notebooks

| Notebook | Description |
|----------|-------------|
| **OutlierAnalysis.ipynb** | Introduces outliers, explains their causes and impact, and demonstrates outlier detection using **Z-Score** and **Interquartile Range (IQR)** methods with Python and NumPy. |
| **InterQuartileRange_Trimming&Capping_Outlier.ipynb** | Demonstrates how to handle outliers detected using the **IQR** method through **Trimming** (removing outliers) and **Capping** (replacing extreme values with boundary values). Includes practical implementation and comparison of both techniques. |

## Technologies Used

- Python
- NumPy
- Matplotlib

## Learning Outcomes

After completing these notebooks, you will be able to:

- Understand what outliers are and why they matter.
- Explain the impact of outliers on statistical analysis and machine learning models.
- Detect outliers using the **Z-Score** method.
- Compute quartiles and the **Interquartile Range (IQR)**.
- Determine lower and upper bounds for identifying outliers.
- Handle outliers using **Trimming** and **Capping** techniques.
- Understand when to remove outliers versus when to cap them.
- Implement outlier detection and treatment techniques in Python.