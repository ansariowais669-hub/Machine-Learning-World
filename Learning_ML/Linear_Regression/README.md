# Linear Regression

This folder contains my practice and implementation of **Linear Regression**, covering **Simple Linear Regression using Scikit-learn**, a **from-scratch implementation using Ordinary Least Squares (OLS)**, **Regression Evaluation Metrics**, and **Multiple Linear Regression**.

The notebooks focus on understanding both the **mathematics behind Linear Regression** and how to **build, evaluate, visualize, and interpret regression models using Python**.

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
* Trained the model on the placement dataset.
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
* Experimented with adding additional features to the dataset.
* Observed how adding an **irrelevant feature** can decrease the Adjusted R² score.
* Observed how adding a **relevant feature** can improve both R² and Adjusted R².
* Understood why Adjusted R² is useful when working with multiple features.

---

### 4. `LR_MultipleLinearRegression.ipynb`

**What I covered:**

* Introduced the concept of **Multiple Linear Regression**, where multiple input features are used to predict a continuous target.
* Generated a synthetic regression dataset using Scikit-learn's `make_regression`.
* Created a dataset with:
  * **100 samples**
  * **2 features**
  * **2 informative features**
  * **1 target**
  * Added noise to the generated data.
* Converted the generated NumPy arrays into a **Pandas DataFrame**.
* Explored the generated dataset containing:
  * `feature1`
  * `feature2`
  * `target`
* Visualized the relationship between the two features and target using a **3D scatter plot**.
* Split the dataset into training and testing sets.
* Implemented Multiple Linear Regression using Scikit-learn's `LinearRegression`.
* Trained the model using multiple input features.
* Generated predictions on the test dataset.
* Evaluated the model using:
  * **Mean Absolute Error (MAE)**
  * **Mean Squared Error (MSE)**
  * **R² Score**
* Visualized the fitted Multiple Linear Regression model as a **3D regression plane** using Plotly.
* Understood the geometric representation of Multiple Linear Regression.

The model in the notebook produced the following evaluation results:

| Metric | Score |
|---|---:|
| MAE | 41.9547 |
| MSE | 2266.0802 |
| R² Score | 0.5146 |

The notebook also creates a 3D visualization containing the original data points and the fitted regression surface. :contentReference[oaicite:2]{index=2}

---

## Simple vs Multiple Linear Regression

### Simple Linear Regression

Uses **one independent feature** to predict the target.

The general equation is:

**`y = mx + b`**

Where:

* `y` = predicted target
* `x` = input feature
* `m` = coefficient/slope
* `b` = intercept

---

### Multiple Linear Regression

Uses **multiple independent features** to predict the target.

The general equation is:

**`y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ`**

Where:

* `y` = predicted target
* `b₀` = intercept
* `x₁, x₂, ..., xₙ` = input features
* `b₁, b₂, ..., bₙ` = coefficients corresponding to each feature

In the newly added notebook, two features (`feature1` and `feature2`) are used to predict the target. :contentReference[oaicite:3]{index=3}

---

## Regression Evaluation Metrics

The regression notebooks introduce several important metrics used to evaluate regression models.

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

---

### Adjusted R²

Adjusted R² modifies the R² score by taking the **number of features** and **number of observations** into account.

Unlike R², Adjusted R² does not automatically improve when additional features are added.

This makes Adjusted R² particularly useful when working with **Multiple Linear Regression** and evaluating whether additional features actually improve the model.

---

## 3D Visualization of Multiple Linear Regression

For Simple Linear Regression, the fitted model can be represented as a **line**.

For Multiple Linear Regression with two features, the model can be visualized as a **plane in 3D space**.

The newly added notebook demonstrates this using **Plotly**, with:

* `feature1` → X-axis
* `feature2` → Y-axis
* `target` → Z-axis
* Regression plane → fitted Multiple Linear Regression model

This provides a visual understanding of how the model fits multiple features simultaneously. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

---

## Libraries Used

* **Pandas** — Data loading, manipulation, and exploration
* **NumPy** — Numerical operations and array handling
* **Matplotlib** — Data visualization
* **Plotly** — Interactive 3D visualizations
* **Scikit-learn** — Train-test splitting, Linear Regression, synthetic dataset generation, and regression evaluation metrics

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
* The difference between **Simple Linear Regression** and **Multiple Linear Regression**.
* How Multiple Linear Regression uses multiple independent features to predict a target.
* How to visualize a Multiple Linear Regression model geometrically as a **3D regression plane**.
* How to generate synthetic regression datasets using `make_regression`.
* How to use **Plotly** for interactive 3D regression visualizations.

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
* Understand the difference between **Simple and Multiple Linear Regression**.
* Build a **Multiple Linear Regression** model using multiple features.
* Interpret the coefficients of a Multiple Linear Regression model.
* Visualize a two-feature regression model using a **3D regression plane**.
* Use Python libraries such as **NumPy, Pandas, Matplotlib, Plotly, and Scikit-learn** for regression tasks.
* Develop a stronger understanding of both **model building and model evaluation** in Linear Regression.

---

## Progression

The notebooks in this folder follow a gradual progression:

**Simple Linear Regression**  
↓  
**Linear Regression from Scratch using OLS**  
↓  
**Regression Evaluation Metrics**  
↓  
**Multiple Linear Regression**  
↓  
**3D Visualization of Regression Plane**

This progression helps build an understanding of Linear Regression from its **basic implementation and mathematical foundations** to **model evaluation and multi-feature regression**.