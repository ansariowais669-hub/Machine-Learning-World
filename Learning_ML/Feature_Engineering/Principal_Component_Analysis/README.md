# Principal Component Analysis

This folder contains my implementation and understanding of **Principal Component Analysis (PCA)**, including both a practical application using MNIST and a step-by-step implementation of PCA from its mathematical foundations.

## Notebooks

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

#### Outcome

Through this notebook, I learned how PCA can reduce a high-dimensional dataset while retaining most of its important information.

On the MNIST dataset, the original **784 features** were reduced to a much smaller number of principal components. I also compared KNN performance before and after PCA and learned how explained variance can be used to decide how many components to retain.

The main takeaway was that **PCA helps reduce dimensionality, computational complexity, and makes high-dimensional data easier to visualize while preserving useful information.**

---

### `PCA_step_by_step.ipynb`

This notebook focuses on understanding **how PCA works internally by implementing its main mathematical steps manually** instead of directly using the PCA implementation from Scikit-Learn.

The notebook covers the following topics:

- **Creating a Sample Dataset**
  - Generating a synthetic 3-dimensional dataset using NumPy
  - Creating two classes with different distributions
  - Creating the target variable

- **3D Data Visualization**
  - Visualizing the original 3-dimensional dataset
  - Using Plotly for interactive 3D visualization

- **Standard Scaling**
  - Applying `StandardScaler`
  - Standardizing the features before performing PCA

- **Covariance Matrix**
  - Calculating the covariance matrix using `np.cov()`
  - Understanding how covariance represents relationships between features

- **Eigenvalues and Eigenvectors**
  - Calculating eigenvalues and eigenvectors using `np.linalg.eig()`
  - Understanding eigenvalues as the amount of variance represented by each principal direction
  - Understanding eigenvectors as the principal directions of the data

- **Visualizing Eigenvectors**
  - Plotting the eigenvectors in 3D
  - Visualizing the principal directions relative to the data

- **Selecting Principal Components**
  - Selecting the top eigenvectors
  - Using the top 2 eigenvectors as the principal components

- **Projecting Data onto Principal Components**
  - Using the dot product to transform the original data
  - Reducing the data from **3 dimensions to 2 dimensions**
  - Creating the transformed dataset containing `PC1` and `PC2`

- **2D Visualization After PCA**
  - Visualizing the transformed data in 2 dimensions
  - Observing the separation of the two classes after dimensionality reduction

#### Outcome

This notebook helped me understand **what happens behind the scenes when PCA is applied**.

Instead of treating PCA as a black-box Scikit-Learn function, I implemented its major steps:

**Standard Scaling → Covariance Matrix → Eigenvalues & Eigenvectors → Selecting Principal Components → Data Transformation**

The original dataset containing **3 features** was successfully transformed into **2 principal components (`PC1` and `PC2`)**, providing a practical understanding of how PCA performs dimensionality reduction mathematically.