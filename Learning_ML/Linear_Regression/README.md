# Linear Regression

This folder contains my practice and implementation of **Simple Linear Regression**, covering both the Scikit-learn approach and a **from-scratch implementation using Ordinary Least Squares (OLS)**.

## Notebooks

### 1. `LR_Simple_LinearRegression.ipynb`

**What I covered:**

* Loaded and explored a placement dataset using **Pandas**.
* Used **CGPA** as the feature and **Package** as the target.
* Visualized the relationship between CGPA and Package.
* Split the dataset into training and testing sets using `train_test_split`.
* Implemented Simple Linear Regression using Scikit-learn's `LinearRegression`.
* Trained the model and generated predictions.
* Visualized the **regression line** and data points.
* Understood the **coefficient (slope)** and **intercept** of the model.
* Connected the implementation with the equation:

  **`y = mx + b`**

### 2. `LR_SimpleLinearRegression_OLS_scratch.ipynb`

**What I covered:**

* Implemented a **Simple Linear Regression model from scratch** without using Scikit-learn's regression model.
* Created a custom `LR` class with `fit()` and `predict()` methods.
* Derived and implemented the **OLS formulas** for calculating:

  * **Slope (`m`)**
  * **Intercept (`b`)**
* Used NumPy arrays for numerical calculations.
* Trained the custom model on the same placement dataset.
* Generated predictions using the custom `predict()` method.
* Understood how the regression model internally calculates its parameters instead of relying on a pre-built library implementation.

## Libraries Used

* **Pandas** — Data loading and manipulation
* **NumPy** — Numerical operations and array handling
* **Matplotlib** — Data visualization
* **Scikit-learn** — Train-test splitting and built-in Linear Regression implementation

## What I Learned

* How **Simple Linear Regression** models the relationship between one input feature and a continuous target.
* How the **slope (`m`)** and **intercept (`b`)** define the regression line.
* How to train and use a Linear Regression model with Scikit-learn.
* How Linear Regression works **internally using Ordinary Least Squares (OLS)**.
* How to calculate the slope and intercept mathematically.
* How to implement a regression model **from scratch using Python and NumPy**.
* How a custom implementation compares conceptually with a library implementation.

## Learning Outcomes

After completing these notebooks, I can:

* Understand the intuition and mathematics behind **Simple Linear Regression**.
* Build a Linear Regression model using **Scikit-learn**.
* Implement Simple Linear Regression **from scratch**.
* Understand and apply the **Ordinary Least Squares (OLS)** method.
* Calculate and interpret the **slope and intercept** of a regression line.
* Prepare data using a train-test split for a regression problem.
* Make predictions using both **library-based and custom models**.
* Understand what happens behind the scenes when using a pre-built Linear Regression implementation.
