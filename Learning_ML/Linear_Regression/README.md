# Linear Regression

This folder contains my practice and implementation of **Linear Regression**, covering **Simple Linear Regression using Scikit-learn**, **from-scratch implementation using Ordinary Least Squares (OLS)**, **Regression Evaluation Metrics**, **Multiple Linear Regression using Scikit-learn**, **Multiple Linear Regression from scratch (Normal Equation)**, **Assumptions of Linear Regression**, **Batch Gradient Descent from scratch**, and **Stochastic Gradient Descent (SGD) from scratch & Scikit-learn**[cite: 1].

The notebooks focus on understanding both the **mathematics behind Linear Regression** and how to **build, evaluate, visualize, implement, and validate regression models using Python**[cite: 1].

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
* Used Scikit-learn's **Diabetes dataset** through `load_diabetes()`[cite: 1].
* Loaded the feature matrix `X` and target variable `y`[cite: 1].
* Split the dataset into training and testing sets using `train_test_split`.
* First implemented Multiple Linear Regression using Scikit-learn's `LinearRegression` as a reference model.
* Trained the Scikit-learn model and generated predictions on the test dataset.
* Evaluated the Scikit-learn model using the **R² Score**.
* Examined the learned **coefficients** and **intercept**.

#### Multiple Linear Regression from Scratch
* Created a custom `MLR` class to implement Multiple Linear Regression.
* Defined the model with:
  * `fit()` method for calculating model parameters.
  * `predict()` method for generating predictions.
* Added a column of ones to the feature matrix to incorporate the **intercept term**.
* Implemented the **Normal Equation** using NumPy:

  **`β = (XᵀX)⁻¹Xᵀy`**

* Used matrix operations such as `np.insert()`, `np.dot()`, and `np.linalg.inv()`.
* Extracted the first value of the calculated parameter vector as the **intercept**.
* Stored the remaining values as the **coefficients** of the input features.
* Used the learned coefficients and intercept to generate predictions manually.
* Evaluated the custom Multiple Linear Regression model using the **R² Score**.
* Printed and compared the learned **coefficients** and **intercept** of the custom implementation.

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
* Observed that higher VIF values indicate stronger multicollinearity (values **greater than 5** indicate Multicollinearity).

#### 3. Normality of Residuals
* Studied the **Normality of Residuals** assumption.
* Understood the importance of examining the distribution of residuals when validating a Linear Regression model.

#### 4. Homoscedasticity
* Studied **Homoscedasticity**, which refers to the residuals having a relatively uniform spread across predicted/independent values.

#### 5. No Autocorrelation of Residuals
* Studied the assumption of **No Autocorrelation of Residuals**.
* Understood that residuals should not exhibit a systematic relationship with one another.

---

### 7. `LR_GradientDescent_from_Scratch.ipynb`

**What I covered:**
This notebook focuses on understanding and implementing **Linear Regression using Batch Gradient Descent from scratch**.

#### Dataset Generation & Scikit-learn Model
* Generated a synthetic regression dataset using Scikit-learn's `make_regression` (100 samples, 1 feature).
* Trained Scikit-learn's `LinearRegression` as a benchmark.
* Used **10-fold cross-validation** with the R² scoring metric.

#### Gradient Descent from Scratch
* Created a custom `GDRegressor` class to implement Linear Regression using **Gradient Descent**.
* Defined parameters: `learning_rate` and `epochs`.
* Updated slope (`m`) and intercept (`b`) using their full-dataset derivatives:
  
  **`loss_slope_b = -2 × Σ(y - mx - b)`**
  
  **`loss_slope_m = -2 × Σ((y - mx - b)x)`**

#### Effect of Learning Rate and Epochs
* Experimented with hyperparameter tuning to avoid divergence and achieve stable convergence near Scikit-learn results.

---

### 8. `LR_SGDRegressor_from_Scratch.ipynb`

**What I covered:**
This notebook covers the step-by-step implementation of **Stochastic Gradient Descent (SGD) for Multiple Linear Regression from scratch** and compares it directly with Scikit-learn's built-in `SGDRegressor`[cite: 1].

#### Dataset & Reference Linear Regression
* Loaded the **Diabetes Dataset** (`442` samples, `10` features) using Scikit-learn[cite: 1].
* Applied a `80-20` train-test split with `random_state=2`[cite: 1].
* Trained a reference Scikit-learn `LinearRegression` model to extract reference coefficients and intercept[cite: 1].

#### Custom `SGDRegressor` Implementation (From Scratch)
* Created a custom class `SGDRegressor` with configurable `learning_rate` and `epochs`[cite: 1].
* Initialized the intercept (`0`) and coefficients (`np.ones(n_features)`)[cite: 1].
* Implemented stochastic updates: for each epoch, iterating over dataset samples selecting random indices (`np.random.randint`)[cite: 1]:
  * Calculated predicted single-point target `y_hat` using vector dot product:
    
    **`y_hat = Xᵢ · W + b`**[cite: 1]
    
  * Derived gradients per instance for intercept and coefficients:
    
    **`∂L/∂b = -2 × (yᵢ - y_hat)`**[cite: 1]
    
    **`∂L/∂W = -2 × (yᵢ - y_hat) × Xᵢ`**[cite: 1]
    
  * Updated parameters in real-time per sample[cite: 1].
* Calculated total training runtime using Python's `time` module[cite: 1].
* Achieved an R² Score of **~0.418** on the test split[cite: 1].

#### Scikit-learn `SGDRegressor` Comparison
* Used `sklearn.linear_model.SGDRegressor` configured with constant learning rate (`learning_rate='constant'`, `eta0=0.01`, `max_iter=100`)[cite: 1].
* Evaluated predictions against the custom class, obtaining a comparable R² Score of **~0.432**[cite: 1].

---

## Simple vs Multiple Linear Regression

### Simple Linear Regression
Uses **one independent feature** to predict the target.

**`y = mx + b`**

---

### Multiple Linear Regression
Uses **multiple independent features** to predict the target[cite: 1].

**`y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ`**

The `LR_MultipleLinearRegression.ipynb` notebook demonstrates Multiple Linear Regression using two features, while `LR_MultipleLR_Scratch.ipynb` and `LR_SGDRegressor_from_Scratch.ipynb` extend the implementation to the **10-feature Diabetes dataset**[cite: 1].

---

## Linear Regression Optimization Approaches

This folder highlights three fundamental methods to compute or approximate regression parameters:

### 1. Analytical Approach
* **Ordinary Least Squares (OLS):** Used in `LR_SimpleLinearRegression_OLS_scratch.ipynb` for single-variable closed-form updates.
* **Normal Equation:** Used in `LR_MultipleLR_Scratch.ipynb` via matrix inverse:

  **`β = (XᵀX)⁻¹Xᵀy`**

### 2. Iterative Optimization Approach
* **Batch Gradient Descent:** Used in `LR_GradientDescent_from_Scratch.ipynb`, updating weights using gradients calculated over the *entire dataset* in each iteration.
* **Stochastic Gradient Descent (SGD):** Used in `LR_SGDRegressor_from_Scratch.ipynb`, updating weights *sample-by-sample* (`1` observation at a time), making it computationally efficient for massive datasets[cite: 1].

---

## Assumptions of Linear Regression

1. **Linear Relationship:** Linear dependence between features and target.
2. **No Multicollinearity:** Predictors are not strongly correlated (checked via **VIF < 5**).
3. **Normality of Residuals:** Residual errors follow a Gaussian distribution.
4. **Homoscedasticity:** Constant variance of residual errors.
5. **No Autocorrelation of Residuals:** Independent residuals across observations.

---

## Regression Evaluation Metrics

* **Mean Absolute Error (MAE):** Average magnitude of absolute errors.
* **Mean Squared Error (MSE):** Average of squared errors (penalizes large outliers).
* **Root Mean Squared Error (RMSE):** Square root of MSE, returned in target units.
* **R² Score:** Variance explained ratio relative to a baseline mean model[cite: 1].
* **Adjusted R² Score:** Penalizes addition of non-informative noise features.

---

## Libraries Used

* **Pandas** — Data loading, manipulation, and exploration
* **NumPy** — Numerical operations, array manipulation, dot products, and matrix operations[cite: 1]
* **Matplotlib** — 2D visualizations and regression line plotting
* **Plotly** — Interactive 3D surface and regression plane visualizations
* **Scikit-learn** — Machine learning algorithms (`LinearRegression`, `SGDRegressor`), metrics (`r2_score`), datasets (`load_diabetes`, `make_regression`), and splitting (`train_test_split`)[cite: 1]
* **Statsmodels** — Statistical diagnostics and VIF computation
* **Time** — Computational runtime benchmarking[cite: 1]

---

## What I Learned

* Difference between analytical exact solvers (**OLS, Normal Equation**) and iterative numerical approximations (**Batch GD, SGD**)[cite: 1].
* How to manually code stochastic parameter updates per sample instance using vector operations (`np.dot`)[cite: 1].
* Practical trade-offs of learning rates (`eta0`), epochs (`max_iter`), and convergence warnings in iterative algorithms[cite: 1].
* Implementing multi-variable custom estimators in Python with standardized `.fit()` and `.predict()` methods[cite: 1].
* Benchmarking custom linear algebra code against battle-tested Scikit-learn standard classes (`LinearRegression`, `SGDRegressor`)[cite: 1].

---

## Progression

**Simple Linear Regression**  
↓  
**Simple Linear Regression from Scratch (OLS)**  
↓  
**Regression Evaluation Metrics**  
↓  
**Multiple Linear Regression using Scikit-learn**  
↓  
**Multiple Linear Regression from Scratch (Normal Equation)**  
↓  
**Assumptions of Linear Regression**  
↓  
**Batch Gradient Descent from Scratch**  
↓  
**Stochastic Gradient Descent (SGD) from Scratch & Scikit-learn**[cite: 1]  
↓  
**3D Visualization of Regression Planes**