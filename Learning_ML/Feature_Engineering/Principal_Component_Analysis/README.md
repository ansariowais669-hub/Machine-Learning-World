# Principal Component Analysis

This folder contains my implementation and understanding of **Principal Component Analysis (PCA)**.

## Notebook

### `PCA_Practical_SKLearn.ipynb`

The notebook covers the following topics:

- **Introduction to PCA**
  - Need for dimensionality reduction
  - High-dimensional data and the Curse of Dimensionality

- **KNN Without PCA**
  - Working with the MNIST dataset
  - Training a KNN classifier on the original 784 features
  - Measuring training and prediction performance

- **Feature Scaling**
  - Applying `StandardScaler`
  - Understanding why scaling is important before PCA

- **Applying PCA**
  - Implementing PCA using `sklearn.decomposition.PCA`
  - Understanding `n_components`
  - Transforming high-dimensional data into a lower-dimensional space

- **Choosing the Number of Components**
  - Explained variance
  - Explained variance ratio
  - Cumulative explained variance
  - Selecting an appropriate number of principal components

- **Understanding PCA Internals**
  - Principal components / eigenvectors
  - Eigenvalues
  - `components_`
  - `explained_variance_`
  - `explained_variance_ratio_`

- **PCA with KNN**
  - Training KNN after dimensionality reduction
  - Comparing model performance with and without PCA

- **PCA Visualization**
  - Reducing MNIST data to **2 dimensions**
  - Reducing MNIST data to **3 dimensions**
  - Visualizing the transformed data

## Outcome

Through this notebook, I learned how PCA can reduce a high-dimensional dataset while retaining most of its important information.

On the MNIST dataset, the original **784 features** were reduced to a much smaller number of principal components. I also compared KNN performance before and after PCA and learned how explained variance can be used to decide how many components to retain.

The main takeaway was that **PCA helps reduce dimensionality, computational complexity, and makes high-dimensional data easier to visualize while preserving useful information.**