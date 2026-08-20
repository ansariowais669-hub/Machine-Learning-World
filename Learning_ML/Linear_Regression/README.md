# Linear Regression

This folder contains my practice and implementation of **Linear Regression**, covering **Simple Linear Regression using Scikit-learn**, **from-scratch implementation using Ordinary Least Squares (OLS)**, **Regression Evaluation Metrics**, **Multiple Linear Regression using Scikit-learn**, **Multiple Linear Regression from scratch**, **Assumptions of Linear Regression**, and **Linear Regression using Gradient Descent from scratch**.

The notebooks focus on understanding both the **mathematics behind Linear Regression** and how to **build, evaluate, visualize, implement, and validate regression models using Python**.

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

| Metric   |     Score |
| -------- | --------: |
| MAE      |   41.9547 |
| MSE      | 2266.0802 |
| R² Score |    0.5146 |

The notebook also creates a 3D visualization containing the original data points and the fitted regression surface.

---

### 5. `LR_MultipleLR_Scratch.ipynb`

**What I covered:**

* Implemented **Multiple Linear Regression from scratch** without directly using Scikit-learn's regression model.
* Used Scikit-learn's **Diabetes dataset** through `load_diabetes()`.
* Loaded the feature matrix `X` and target variable `y`.
* Split the dataset into training and testing sets using `train_test_split`.
* First implemented Multiple Linear Regression using Scikit-learn's `LinearRegression` as a reference model.
* Trained the Scikit-learn model and generated predictions on the test dataset.
* Evaluated the Scikit-learn model using the **R² Score**.
* Examined the learned:

  * **Coefficients**
  * **Intercept**

#### Multiple Linear Regression from Scratch

* Created a custom `MLR` class to implement Multiple Linear Regression.

* Defined the model with:

  * `fit()` method for calculating model parameters.
  * `predict()` method for generating predictions.

* Added a column of ones to the feature matrix to incorporate the **intercept term**.

* Implemented the **Normal Equation** using NumPy:

  **`β = (XᵀX)⁻¹Xᵀy`**

* Used matrix operations such as:

  * `np.insert()`
  * `np.dot()`
  * `np.linalg.inv()`

* Extracted the first value of the calculated parameter vector as the **intercept**.

* Stored the remaining values as the **coefficients** of the input features.

* Used the learned coefficients and intercept to generate predictions manually.

* Evaluated the custom Multiple Linear Regression model using the **R² Score**.

* Printed and compared the learned **coefficients** and **intercept** of the custom implementation.

This notebook helped connect the mathematical formulation of Multiple Linear Regression with an actual Python implementation from scratch.

---

### 6. `LR_Assumptions_of_LR.ipynb`

**What I covered:**

This notebook focuses on understanding the important **assumptions of Linear Regression** that should be considered when building and interpreting regression models.

#### 1. Linear Relationship

* Studied the assumption that there should be a **linear relationship** between the independent variables and the dependent variable.
* Understood why Linear Regression works best when the relationship between predictors and the target can be reasonably represented using a linear function.

#### 2. Multicollinearity

* Studied **Multicollinearity**, where independent features are strongly related to each other.
* Used the **Variance Inflation Factor (VIF)** concept to identify multicollinearity.
* Observed that higher VIF values indicate stronger multicollinearity.
* In the notebook, values **greater than 5** are considered an indication of Multicollinearity.

#### 3. Normality of Residuals

* Studied the **Normality of Residuals** assumption.
* Understood the importance of examining the distribution of residuals when validating a Linear Regression model.

#### 4. Homoscedasticity

* Studied **Homoscedasticity**, which refers to the residuals having a relatively uniform spread.
* Observed the idea that the spread should remain approximately consistent across the range of predicted/independent values.

#### 5. No Autocorrelation of Residuals

* Studied the assumption of **No Autocorrelation of Residuals**.
* Understood that residuals should not exhibit a systematic relationship with one another.
* This assumption is particularly important when observations have an ordered or time-dependent structure.

---

### 7. `LR_GradientDescent_from_Scratch.ipynb`

**What I covered:**

This notebook focuses on understanding and implementing **Linear Regression using Gradient Descent from scratch**.

#### Dataset Generation

* Generated a synthetic regression dataset using Scikit-learn's `make_regression`.
* Created a dataset with:

  * **100 samples**
  * **1 feature**
  * **1 informative feature**
  * **1 target**
  * Added noise to the generated data.
* Visualized the generated data using a scatter plot.

#### Scikit-learn Reference Model

* Implemented Linear Regression using Scikit-learn's `LinearRegression`.
* Trained the model on the generated dataset.
* Examined the learned:

  * **Coefficient (`m`)**
  * **Intercept (`b`)**
* Used **10-fold cross-validation** with the R² scoring metric.
* Calculated the mean cross-validation R² score as a reference for the custom implementation.

#### Gradient Descent from Scratch

* Created a custom `GDRegressor` class to implement Linear Regression using **Gradient Descent**.
* Defined the model with:

  * `learning_rate`
  * `epochs`
* Initialized the coefficient and intercept manually.
* Implemented a `fit()` method to iteratively update the model parameters.
* Implemented a `predict()` method to generate predictions.

The model follows the basic Linear Regression equation:

**`y = mx + b`**

The parameters `m` and `b` are updated iteratively using their respective gradients.

#### Gradient Calculation

The notebook calculates the slope of the loss function with respect to the intercept:

**`loss_slope_b = -2 × Σ(y - mx - b)`**

And the slope of the loss function with respect to the coefficient:

**`loss_slope_m = -2 × Σ((y - mx - b)x)`**

The parameters are then updated using:

**`b = b - learning_rate × loss_slope_b`**

**`m = m - learning_rate × loss_slope_m`**

This demonstrates how Gradient Descent gradually updates the parameters in order to minimize the regression loss.

#### Effect of Learning Rate and Epochs

* Experimented with different **learning rates** and **numbers of epochs**.
* Observed that a learning rate that is too high can lead to very large parameter values.
* Reduced the learning rate to obtain more stable parameter updates.
* Increased the number of epochs and observed that the learned parameters became closer to those obtained from Scikit-learn.
* Understood the importance of choosing an appropriate **learning rate** and **number of iterations**.

#### Comparing Gradient Descent with Scikit-learn

* Split the dataset into training and testing sets using `train_test_split`.
* Trained Scikit-learn's `LinearRegression` model on the training data.
* Generated predictions on the test set.
* Calculated the **R² Score**.
* Trained the custom `GDRegressor` on the same training data.
* Generated predictions using the custom Gradient Descent implementation.
* Calculated its **R² Score**.
* Observed that the custom Gradient Descent implementation produced **nearly the same results** as Scikit-learn's Linear Regression.

This notebook helped connect the mathematical idea of **Gradient Descent** with an actual implementation of Linear Regression from scratch.

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

The `LR_MultipleLinearRegression.ipynb` notebook demonstrates Multiple Linear Regression using two features, while `LR_MultipleLR_Scratch.ipynb` extends the implementation to the **multiple-feature Diabetes dataset**.

---

## Linear Regression from Scratch

This folder now contains multiple approaches to implementing Linear Regression without relying completely on Scikit-learn.

### Ordinary Least Squares

The `LR_SimpleLinearRegression_OLS_scratch.ipynb` notebook implements Simple Linear Regression using **Ordinary Least Squares (OLS)**.

### Normal Equation

The `LR_MultipleLR_Scratch.ipynb` notebook implements Multiple Linear Regression using the **Normal Equation**:

**`β = (XᵀX)⁻¹Xᵀy`**

### Gradient Descent

The `LR_GradientDescent_from_Scratch.ipynb` notebook implements Linear Regression using **Gradient Descent**.

This provides two different approaches for understanding how regression parameters can be learned:

**Analytical Approach**

`Normal Equation`

↓

**Iterative Optimization Approach**

`Gradient Descent`

The Gradient Descent implementation updates the model parameters repeatedly using the gradients of the loss function until the parameters approach suitable values.

---

## Multiple Linear Regression from Scratch

The `LR_MultipleLR_Scratch.ipynb` notebook demonstrates how Multiple Linear Regression can be implemented mathematically without relying on Scikit-learn's `LinearRegression` class.

For a dataset with multiple features, the model can be represented in matrix form as:

**`y = Xβ`**

The parameters can be calculated using the **Normal Equation**:

**`β = (XᵀX)⁻¹Xᵀy`**

Where:

* `X` = feature matrix
* `Xᵀ` = transpose of the feature matrix
* `β` = vector containing the model coefficients and intercept
* `y` = target vector

To include the intercept, a column of ones is added to the feature matrix.

The custom implementation follows this process:

```text
Feature Matrix
      ↓
Add column of 1s
      ↓
Calculate XᵀX
      ↓
Calculate (XᵀX)⁻¹
      ↓
Calculate Xᵀy
      ↓
β = (XᵀX)⁻¹Xᵀy
      ↓
Extract intercept and coefficients
      ↓
Generate predictions
```

This provides a direct connection between the **mathematical Normal Equation** and the actual implementation of Multiple Linear Regression.

---

## Linear Regression using Gradient Descent

Gradient Descent is an **iterative optimization algorithm** that can be used to learn the parameters of a Linear Regression model.

For Simple Linear Regression:

**`y = mx + b`**

where:

* `m` = coefficient/slope
* `b` = intercept

Instead of calculating the parameters directly using the Normal Equation, Gradient Descent starts with initial values and repeatedly updates `m` and `b` in the direction that reduces the loss.

### Parameter Updates

The general update rule is:

**`parameter = parameter - learning_rate × gradient`**

For the intercept:

**`b = b - learning_rate × loss_slope_b`**

For the coefficient:

**`m = m - learning_rate × loss_slope_m`**

The process is repeated for a specified number of **epochs**.

### Learning Rate

The **learning rate** controls the size of each parameter update.

* A learning rate that is too high can cause unstable or excessively large updates.
* A smaller learning rate provides more controlled updates.
* The learning rate must be chosen appropriately for the optimization process.

The notebook demonstrates this behavior by experimenting with different learning rates.

### Epochs

An **epoch** represents one complete iteration through the training process.

Increasing the number of epochs gives the model more opportunities to update its parameters and approach suitable values.

The notebook demonstrates that after using an appropriate learning rate and sufficient epochs, the custom Gradient Descent model obtains parameter values close to those produced by Scikit-learn's Linear Regression.

---

## Assumptions of Linear Regression

For Linear Regression models to provide reliable and meaningful results, several important assumptions should be considered.

### 1. Linear Relationship

There should be a reasonably **linear relationship between the independent variables and the dependent variable**.

---

### 2. No Multicollinearity

The independent variables should not be highly correlated with each other.

High multicollinearity can make coefficient estimates unstable and difficult to interpret.

The notebook uses the **VIF (Variance Inflation Factor)** concept, where higher values indicate a greater degree of multicollinearity.

---

### 3. Normality of Residuals

The residuals should approximately follow a **normal distribution**.

Residuals are the differences between the actual and predicted values:

**`Residual = Actual − Predicted`**

---

### 4. Homoscedasticity

The residuals should have a relatively **constant spread** across the range of predicted values or independent variables.

A changing or funnel-shaped residual spread indicates **heteroscedasticity**.

---

### 5. No Autocorrelation of Residuals

Residuals should be **independent of one another** and should not exhibit systematic patterns or correlations.

This assumption becomes especially important for datasets where observations have a natural order, such as time-series data.

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

The `LR_MultipleLinearRegression.ipynb` notebook demonstrates this using **Plotly**, with:

* `feature1` → X-axis
* `feature2` → Y-axis
* `target` → Z-axis
* Regression plane → fitted Multiple Linear Regression model

This provides a visual understanding of how the model fits multiple features simultaneously.

---

## Libraries Used

* **Pandas** — Data loading, manipulation, and exploration
* **NumPy** — Numerical operations, array handling, and matrix calculations
* **Matplotlib** — Data visualization
* **Plotly** — Interactive 3D visualizations
* **Scikit-learn** — Dataset generation, train-test splitting, Linear Regression, cross-validation, and regression evaluation
* **Statsmodels** — Statistical analysis and tools for investigating Linear Regression assumptions

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
* How to implement **Multiple Linear Regression from scratch**.
* How the **Normal Equation** is used to calculate the parameters of Multiple Linear Regression.
* How to represent Linear Regression mathematically using **matrices and vectors**.
* How to add an intercept term to a feature matrix.
* How to calculate regression coefficients using NumPy matrix operations.
* How to compare a **custom implementation** with Scikit-learn's implementation.
* How to visualize a two-feature Multiple Linear Regression model as a **3D regression plane**.
* How to generate synthetic regression datasets using `make_regression`.
* How to use the Scikit-learn **Diabetes dataset** for a regression problem.
* Why understanding the **assumptions of Linear Regression** is important before interpreting a model.
* What **Linear Relationship** means in the context of Linear Regression.
* What **Multicollinearity** is and why it can affect regression models.
* How **VIF** can be used to investigate Multicollinearity.
* Why **Normality of Residuals** is an important assumption.
* What **Homoscedasticity** means and why constant residual variance matters.
* Why **No Autocorrelation of Residuals** is important for independent observations.
* How **Gradient Descent** can be used to learn Linear Regression parameters.
* How the **learning rate** controls the size of parameter updates.
* How the number of **epochs** affects the optimization process.
* How to calculate gradients for the coefficient and intercept.
* How to implement a custom `GDRegressor` class using NumPy.
* How to compare a Gradient Descent implementation with Scikit-learn's Linear Regression.
* How to evaluate the Gradient Descent model using **R² Score**.
* How an appropriate learning rate and sufficient epochs can produce results close to Scikit-learn's implementation.
* How model evaluation involves not only measuring performance but also checking whether the assumptions behind the model are reasonably satisfied.
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
* Understand the **Normal Equation** used in Multiple Linear Regression.
* Implement Multiple Linear Regression **from scratch using NumPy**.
* Understand the concept of **Gradient Descent** for Linear Regression.
* Implement Linear Regression using **Gradient Descent from scratch**.
* Understand the role of the **learning rate** and **epochs**.
* Calculate and apply gradients to update Linear Regression parameters.
* Compare Gradient Descent with Scikit-learn's `LinearRegression`.
* Visualize a two-feature regression model using a **3D regression plane**.
* Understand the major **assumptions of Linear Regression**.
* Identify and understand **Multicollinearity**.
* Use **VIF** as a measure for investigating multicollinearity.
* Understand the importance of **Normality of Residuals**.
* Understand and identify **Homoscedasticity**.
* Understand the importance of **No Autocorrelation of Residuals**.
* Use Python libraries such as **NumPy, Pandas, Matplotlib, Plotly, and Scikit-learn** for regression tasks.
* Develop a stronger understanding of **model building, mathematical implementation, optimization, model evaluation, and model validation** in Linear Regression.

---

## Progression

The notebooks in this folder follow a gradual progression:

**Simple Linear Regression**

↓

**Linear Regression from Scratch using OLS**

↓

**Regression Evaluation Metrics**

↓

**Multiple Linear Regression using Scikit-learn**

↓

**Multiple Linear Regression from Scratch**

↓

**Assumptions of Linear Regression**

↓

**Linear Regression using Gradient Descent from Scratch**

↓

**3D Visualization of Regression Plane**

This progression helps build an understanding of Linear Regression from its **basic implementation and mathematical foundations** to **model evaluation, multi-feature regression, matrix-based implementation, model assumptions, iterative optimization using Gradient Descent, and visualization**.
