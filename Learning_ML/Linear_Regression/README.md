# Linear Regression

This folder contains my practice and implementation of **Linear Regression**, covering **Simple Linear Regression using Scikit-learn**, a **from-scratch implementation using Ordinary Least Squares (OLS)**, and **regression model evaluation using different performance metrics**.

The notebooks focus on understanding both the **mathematics behind Linear Regression** and how to **build, evaluate, and interpret regression models using Python**.

---

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

---

### 2. `LR_SimpleLinearRegression_OLS_scratch.ipynb`

**What I covered:**

* Implemented a **Simple Linear Regression model from scratch** without using Scikit-learn's regression model.
* Created a custom `LR` class with `fit()` and `predict()` methods.
* Derived and implemented the **Ordinary Least Squares (OLS)** formulas for calculating:

  * **Slope (`m`)**
  * **Intercept (`b`)**
* Used NumPy arrays for numerical calculations.
* Trained the custom model on the placement dataset.
* Generated predictions using the custom `predict()` method.
* Understood how the regression model internally calculates its parameters instead of relying on a pre-built library implementation.

---

### 3. `LR_Regression_Metrics.ipynb`

**What I covered:**

* Loaded and explored the placement dataset containing **CGPA** and **Package**.
* Built a Linear Regression model using Scikit-learn.
* Split the dataset into training and testing sets using `train_test_split`.
* Generated predictions using the trained regression model.
* Evaluated the model using different **Regression Evaluation Metrics**:

  * **Mean Absolute Error (MAE)**
  * **Mean Squared Error (MSE)**
  * **Root Mean Squared Error (RMSE)**
  * **R² Score**
  * **Adjusted R² Score**
* Compared the usefulness of different regression metrics for evaluating model performance.
* Experimented with adding an additional feature to the dataset.
* Observed how adding an **irrelevant feature** can decrease the Adjusted R² score.
* Observed how adding a **relevant feature** can improve both R² and Adjusted R².
* Understood why Adjusted R² is useful when working with multiple features.

The notebook obtained the following evaluation results for one of the regression experiments:

| Metric            |  Score |
| ----------------- | -----: |
| MAE               | 0.2885 |
| MSE               | 0.1213 |
| RMSE              | 0.3483 |
| R² Score          | 0.7807 |
| Adjusted R² Score | 0.7750 |

## The notebook also experimented with an additional feature and observed an R² score of approximately **0.8164** and an Adjusted R² score of approximately **0.8065**.

## Libraries Used

* **Pandas** — Data loading, manipulation, and exploration
* **NumPy** — Numerical operations and array handling
* **Matplotlib** — Data visualization
* **Scikit-learn** — Train-test splitting, Linear Regression, and regression evaluation metrics

---

## Regression Evaluation Metrics

The third notebook introduces several important metrics used to evaluate regression models.

### Mean Absolute Error (MAE)

Measures the average absolute difference between the actual and predicted values.

**Lower MAE → Better model**

---

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted values.

**Lower MSE → Better model**

Since the errors are squared, larger errors have a greater impact on the metric.

---

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

**Lower RMSE → Better model**

One advantage of RMSE is that it is expressed in the **same units as the target variable**.

---

### R² Score

R², or the **Coefficient of Determination**, measures how well the model explains the variance in the target variable.

A higher R² generally indicates a better fit.

For example, the notebook obtained an R² score of approximately **0.781** in one experiment.

---

### Adjusted R²

Adjusted R² modifies the R² score by taking the **number of features** and **number of observations** into account.

Unlike R², Adjusted R² does not automatically improve when additional features are added.

The notebook demonstrates this experimentally:

* Adding an **irrelevant feature** decreased the Adjusted R² score.
* Adding a **relevant feature** increased the Adjusted R² score.

This makes Adjusted R² particularly useful when evaluating models with multiple features.

---

## What I Learned

* How **Simple Linear Regression** models the relationship between one input feature and a continuous target.
* How the **slope (`m`)** and **intercept (`b`)** define the regression line.
* How to train and use a Linear Regression model with Scikit-learn.
* How Linear Regression works internally using **Ordinary Least Squares (OLS)**.
* How to calculate the slope and intercept mathematically.
* How to implement a regression model **from scratch using Python and NumPy**.
* How to split data into training and testing sets.
* How to generate predictions using a trained regression model.
* How to evaluate regression models using **MAE, MSE, RMSE, and R²**.
* What **Adjusted R²** means and why it is useful.
* How adding features can affect model evaluation.
* Why adding an irrelevant feature can negatively affect **Adjusted R²**.
* Why relevant features can improve model performance.
* How to compare different regression models using appropriate evaluation metrics.

---

## Learning Outcomes

After completing these notebooks, I can:

* Understand the intuition and mathematics behind **Simple Linear Regression**.
* Build a Linear Regression model using **Scikit-learn**.
* Implement Simple Linear Regression **from scratch**.
* Understand and apply the **Ordinary Least Squares (OLS)** method.
* Calculate and interpret the **slope and intercept** of a regression line.
* Prepare data using a **train-test split** for a regression problem.
* Make predictions using both **library-based and custom models**.
* Evaluate regression models using **MAE, MSE, RMSE, and R²**.
* Calculate and interpret **Adjusted R²**.
* Understand the effect of adding relevant and irrelevant features to a regression model.
* Understand what happens behind the scenes when using a pre-built Linear Regression implementation.
* Develop a stronger understanding of both **model building and model evaluation** in Linear Regression.
