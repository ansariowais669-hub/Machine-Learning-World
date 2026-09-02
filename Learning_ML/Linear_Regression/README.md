# Linear, Polynomial & Regularized Regression

This repository contains my practice and hands-on implementations of continuous variable regression models using Python. It covers **Simple Linear Regression (Scikit-learn and from-scratch OLS)**, **Multiple Linear Regression (Scikit-learn and from-scratch Normal Equation)**, **Regression Evaluation Metrics**, **Assumptions of Linear Regression**, **Iterative Optimization (Batch, Stochastic, and Mini-Batch Gradient Descent)**, **Polynomial Regression**, and **Ridge Regression (L2 Regularization)**.

The primary goal of this repository is to bridge theoretical mathematical concepts with practical Python implementations, model training, performance evaluation, diagnostics, and 3D visualization.

---

## Notebooks

### 1. `LR_Simple_LinearRegression.ipynb`
* **Dataset:** Placement dataset (CGPA vs. Package).
* **Concepts:** Exploratory Data Analysis with Pandas, data splitting via `train_test_split`.
* **Implementation:** Built a Simple Linear Regression model using Scikit-learn's `LinearRegression`.
* **Visualization:** Plotted the fitted 2D regression line alongside actual data points.
* **Math Link:** Analyzed model attributes (slope $m$ and intercept $b$) matching the equation:
  
  $$y = mx + b$$

---

### 2. `LR_SimpleLinearRegression_OLS_scratch.ipynb`
* **Concepts:** Custom implementation of Simple Linear Regression from scratch without using machine learning libraries.
* **Implementation:** Created a custom `LR` class with object-oriented `fit()` and `predict()` methods.
* **Math Derived:** Implemented closed-form **Ordinary Least Squares (OLS)** formulas for slope ($m$) and intercept ($b$) using NumPy:

  $$m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad b = \bar{y} - m\bar{x}$$

---

### 3. `LR_Regression_Metrics.ipynb`
* **Dataset:** Placement dataset.
* **Metrics Covered:** Evaluated models using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), $R^2$ Score, and Adjusted $R^2$ Score.
* **Feature Analysis:** Added relevant vs. irrelevant features to demonstrate why Adjusted $R^2$ penalizes non-informative features, preventing artificial score inflation:

  $$\text{Adjusted } R^2 = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - k - 1} \right]$$

---

### 4. `LR_MultipleLinearRegression.ipynb`
* **Dataset:** Synthetic regression dataset (`100` samples, `2` features, added Gaussian noise) generated using Scikit-learn's `make_regression`.
* **Implementation:** Built Multiple Linear Regression using Scikit-learn's `LinearRegression`.
* **Metrics Obtained:**
  * **MAE:** $41.9547$
  * **MSE:** $2266.0802$
  * **$R^2$ Score:** $0.5146$
* **Visualization:** Interactive 3D scatter plots and fitted 3D regression planes generated using Plotly.

---

### 5. `LR_MultipleLR_Scratch.ipynb`
* **Dataset:** Scikit-learn's Diabetes Dataset (`442` samples, `10` features).
* **Implementation:** Created a custom `MLR` class to solve Multiple Linear Regression analytically.
* **Math Derived:** Added a column of ones for the bias/intercept term and implemented the **Normal Equation**:

  $$\beta = (X^T X)^{-1} X^T y$$

* **Validation:** Verified custom matrix operations (`np.dot`, `np.linalg.inv`) directly against Scikit-learn's benchmark parameters and $R^2$ score.

---

### 6. `LR_Assumptions_of_LR.ipynb`
Focused on testing and validating the fundamental assumptions required for linear modeling:
1. **Linear Relationship:** Linear dependence between predictors and target.
2. **Multicollinearity:** Evaluated using the **Variance Inflation Factor (VIF)**. Values $> 5$ indicate significant multicollinearity.
3. **Normality of Residuals:** Analyzed residual error distribution graphs.
4. **Homoscedasticity:** Checked for uniform variance of residual errors across predictions.
5. **No Autocorrelation of Residuals:** Ensured independence between residual errors.

---

### 7. `LR_GradientDescent_from_Scratch.ipynb`
* **Concepts:** First-principles implementation of **Batch Gradient Descent (BGD)**.
* **Implementation:** Developed a custom `GDRegressor` class.
* **Math Derived:** Computed gradients over the entire dataset per epoch to update parameters:

  $$\frac{\partial L}{\partial b} = -2 \sum (y_i - (mx_i + b))$$

  $$\frac{\partial L}{\partial m} = -2 \sum (y_i - (mx_i + b))x_i$$

* **Hyperparameter Tuning:** Evaluated the effect of learning rates ($\eta$) and epoch counts on convergence stability.

---

### 8. `LR_SGDRegressor_from_Scratch.ipynb`
* **Dataset:** Diabetes Dataset (`442` samples, `10` features).
* **Implementation:** Built a custom `SGDRegressor` for **Stochastic Gradient Descent (SGD)** from scratch and benchmarked against `sklearn.linear_model.SGDRegressor`.
* **Update Rule:** Updated weight vectors sample-by-sample ($1$ observation at a time) using randomly chosen indices per epoch:

  $$y_{hat} = X_i \cdot W + b$$
  $$\frac{\partial L}{\partial b} = -2(y_i - y_{hat}), \quad \frac{\partial L}{\partial W} = -2(y_i - y_{hat})X_i$$

* **Results:** Custom class achieved $R^2 \approx 0.418$; Scikit-learn achieved $R^2 \approx 0.432$.

---

### 9. `LR_Mini_Batch_GD_Scratch.ipynb`
* **Dataset:** Diabetes Dataset (`442` samples, `10` features).
* **Implementation:** Vectorized **Mini-Batch Gradient Descent (MBGD)** built from scratch and compared against baseline OLS and Scikit-learn's `SGDRegressor`.
* **Custom MBGD:** Utilized random sub-batches (`batch_size ≈ 7`, `lr = 0.01`, `epochs = 75`) to compute batch gradients:

  $$\frac{\partial L}{\partial W} = -2 \cdot (y_{\text{batch}} - y_{\text{hat}})^T \cdot X_{\text{batch}}$$

* **Partial Fit:** Evaluated streaming mini-batch execution using Scikit-learn's `.partial_fit()` method.
* **Results:** Custom MBGD achieved $R^2 = 0.4472$; Scikit-learn `partial_fit` achieved $R^2 = 0.4315$.

---

### 10. `Polynomial_Regression.ipynb`
* **Dataset:** Non-linear quadratic synthetic dataset ($y = 0.8x^2 + 0.9x + 2 + \text{noise}$).
* **Underfitting Analysis:** Standard Linear Regression achieved a poor $R^2 \approx 0.2691$ due to high bias.
* **Polynomial Features:** Applied feature transformation via `PolynomialFeatures` to capture quadratic boundaries and curved data trends.

---

### 11. `Ridge_Regression.ipynb` *(Newly Added)*
* **Concepts:** Introduction to **L2 Regularization (Ridge Regression)** to prevent model overfitting and mitigate multicollinearity by shrinking coefficient magnitudes.
* **Dataset:** Synthetic 1D regression dataset generated via `make_regression` (`100` samples, noise added, `random_state=13`).
* **Implementation:** Compared standard Scikit-learn `LinearRegression` against `Ridge` regression across different penalty parameter strengths ($\alpha$).
* **Model Benchmark & Parameter Shrinkage:**

  | Model | Penalty Strength ($\alpha$) | Learned Slope ($m$) | Intercept ($b$) |
  | :--- | :--- | :--- | :--- |
  | **Linear Regression** | $\alpha = 0$ | $27.8281$ | $-2.2947$ |
  | **Ridge Regression** | $\alpha = 10$ | $24.9546$ | $-2.1269$ |
  | **Ridge Regression** | $\alpha = 100$ | $12.9344$ | $-1.4248$ |

* **Key Observations:** As $\alpha$ increases, the penalty forces the coefficient values closer to zero, smoothing model variance while reducing sensitivity to noisy data points.

---
### Mathematical Overview

* **Simple Linear Regression:** 
  $$y = mx + b$$
* **Multiple Linear Regression:** 
  $$y = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_n x_n$$
* **Polynomial Regression:** 
  $$y = b_0 + b_1 x + b_2 x^2 + \dots + b_n x^n$$
* **Ridge Regression (L2 Loss Minimization):** 
  $$\text{Cost} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} w_j^2$$

---

## Optimization Approaches Summary

1. **Analytical Approaches (Exact Closed-Form):**
   * **Ordinary Least Squares (OLS):** Used for single-variable direct mathematical parameter resolution.
   * **Normal Equation:** Matrix inversion $\beta = (X^T X)^{-1} X^T y$ for multi-feature optimization.

2. **Iterative Numerical Optimization:**
   * **Batch Gradient Descent:** Computes updates using the entire dataset per epoch (stable, high memory usage).
   * **Stochastic Gradient Descent (SGD):** Computes updates observation-by-observation (fast, noisy updates).
   * **Mini-Batch Gradient Descent (MBGD):** Computes updates over subset blocks (`batch_size`), balancing vector computational speed and update stability.

3. **Regularized Shrinkage Optimization:**
   * **Ridge (L2):** Adds a squared penalty term ($\alpha$) to constrain weights, reducing model complexity and overfitting.

---

## Regression Evaluation Metrics

* **Mean Absolute Error (MAE):** Measures average absolute deviation magnitude.
* **Mean Squared Error (MSE):** Squares deviations, severely penalizing larger outliers.
* **Root Mean Squared Error (RMSE):** Square root of MSE, expressing error back in original target variable units.
* **$R^2$ Score:** Indicates the proportion of target variance explained by model features relative to a baseline mean model.
* **Adjusted $R^2$ Score:** Modifies $R^2$ by penalizing non-informative features added to the model.

---

## Assumptions of Linear Regression

1. **Linearity:** Independent variables have a linear relationship with the target.
2. **No Multicollinearity:** Independent predictors are not highly correlated ($VIF < 5$).
3. **Normality of Residuals:** Residual errors follow a zero-mean Gaussian distribution.
4. **Homoscedasticity:** Residual variance remains constant across all fitted values.
5. **No Autocorrelation:** Residuals are independent of each other across time or observations.

---

## Libraries & Tools Used

* **Pandas** — Data manipulation, CSV parsing, and DataFrame transformations.
* **NumPy** — Linear algebra computations, matrix inversions, dot products, and array manipulations.
* **Matplotlib** — 2D scatter plots, residual distribution plots, and line visualizations.
* **Plotly** — Interactive 3D scatter plots and surface regression planes.
* **Scikit-learn** — Estimators (`LinearRegression`, `Ridge`, `SGDRegressor`), dataset generators (`make_regression`, `load_diabetes`), evaluation metrics (`r2_score`), and feature transformers (`PolynomialFeatures`, `StandardScaler`).
* **Statsmodels** — Diagnostic statistical tools and VIF calculation.

---

---

## Key Learnings & Takeaways

* Applied exact closed-form solvers (**OLS, Normal Equation**) alongside approximate iterative solvers (**Batch, Stochastic, and Mini-Batch GD**).
* Mastered low-level vectorized NumPy updates for stochastic and mini-batch gradient calculations.
* Leveraged Scikit-learn's `.partial_fit()` API for online and streaming mini-batch workflows.
* Addressed non-linear structural patterns using polynomial feature expansions.
* Observed the direct impact of **L2 Regularization (Ridge)** in shrinking weight magnitudes ($\alpha = 0 \rightarrow \alpha = 100$) to control model variance and prevent overfitting.
* Benchmarked custom Python linear algebra implementations directly against optimized Scikit-learn estimators.