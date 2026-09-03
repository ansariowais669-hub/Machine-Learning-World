# Linear, Polynomial & Regularized Regression

This repository contains my practice and hands-on implementations of **regression algorithms using Python**, with an emphasis on understanding the mathematical foundations behind each model and implementing them both through **Scikit-learn** and **from scratch** where applicable.

The repository progressively explores **Simple Linear Regression, Multiple Linear Regression, Regression Evaluation Metrics, Linear Regression Assumptions, Batch Gradient Descent, Stochastic Gradient Descent, Mini-Batch Gradient Descent, Polynomial Regression, Ridge Regression (L2 Regularization), and Ridge Regression using Gradient Descent**.

The primary goal of this repository is to bridge **theoretical mathematical concepts with practical machine learning implementations**, including model training, optimization, evaluation, diagnostics, regularization, visualization, and comparison between different approaches.

---

## Notebooks

### 1. `LR_Simple_LinearRegression.ipynb`

* **Dataset:** Placement dataset containing **CGPA vs. Package**.
* **Concepts:** Exploratory Data Analysis using Pandas and splitting the dataset using `train_test_split`.
* **Implementation:** Built a Simple Linear Regression model using Scikit-learn's `LinearRegression`.
* **Visualization:** Plotted the fitted 2D regression line together with the actual observations.
* **Mathematical Foundation:** Studied the relationship between the learned slope, intercept, and the linear equation:

```text
y = mx + b
```

---

### 2. `LR_SimpleLinearRegression_OLS_scratch.ipynb`

* **Concepts:** Implemented Simple Linear Regression completely from scratch without using a machine learning estimator.
* **Implementation:** Created a custom `LR` class with object-oriented `fit()` and `predict()` methods.
* **Mathematical Foundation:** Implemented the closed-form **Ordinary Least Squares (OLS)** equations using NumPy:

```text
m = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²

b = ȳ - mx̄
```

* **Learning Focus:** Connected the mathematical derivation of linear regression directly with its Python implementation.

---

### 3. `LR_Regression_Metrics.ipynb`

* **Dataset:** Placement dataset.
* **Metrics Covered:** Evaluated regression models using:

  * **Mean Absolute Error (MAE)**
  * **Mean Squared Error (MSE)**
  * **Root Mean Squared Error (RMSE)**
  * **$R^2$ Score**
  * **Adjusted $R^2$ Score**
* **Feature Analysis:** Compared the effect of relevant and irrelevant features on regression performance.
* **Adjusted $R^2$:** Demonstrated how Adjusted $R^2$ penalizes the addition of non-informative features.

---

### 4. `LR_MultipleLinearRegression.ipynb`

* **Dataset:** Synthetic regression dataset containing `100` samples and `2` features, generated using Scikit-learn's `make_regression` with Gaussian noise.
* **Implementation:** Built a Multiple Linear Regression model using Scikit-learn's `LinearRegression`.
* **Model Evaluation:**

  * **MAE:** `41.9547`
  * **MSE:** `2266.0802`
  * **$R^2$ Score:** `0.5146`
* **Visualization:** Used Plotly to create interactive **3D scatter plots** and visualize the fitted **3D regression plane**.
* **Learning Focus:** Extended the concept of Simple Linear Regression from one predictor to multiple predictors.

---

### 5. `LR_MultipleLR_Scratch.ipynb`

* **Dataset:** Scikit-learn's **Diabetes Dataset** containing `442` samples and `10` features.
* **Implementation:** Created a custom `MLR` class to solve Multiple Linear Regression analytically.
* **Mathematical Foundation:** Added a column of ones for the intercept and implemented the **Normal Equation**:

```text
β = (XᵀX)⁻¹Xᵀy
```

* **Validation:** Compared the custom matrix-based implementation against Scikit-learn's implementation and verified the learned parameters and $R^2$ performance.

---

### 6. `LR_Assumptions_of_LR.ipynb`

This notebook focuses on understanding and testing the fundamental assumptions behind Linear Regression.

The major assumptions explored are:

1. **Linearity:** Independent variables should have a linear relationship with the target.
2. **No Multicollinearity:** Predictors should not exhibit problematic levels of correlation.
3. **Normality of Residuals:** Residual errors should approximately follow a normal distribution.
4. **Homoscedasticity:** The variance of residuals should remain approximately constant across fitted values.
5. **No Autocorrelation:** Residual errors should be independent of one another.

* **Diagnostic Tool:** Variance Inflation Factor (**VIF**) is used for evaluating multicollinearity.

---

### 7. `LR_GradientDescent_from_Scratch.ipynb`

* **Concepts:** First-principles implementation of **Batch Gradient Descent (BGD)** for Linear Regression.
* **Implementation:** Developed a custom `GDRegressor` class.
* **Optimization:** Calculated gradients using the complete training dataset during every epoch.
* **Gradient Equations:** Derived the gradients required to update the model parameters.
* **Hyperparameter Analysis:** Studied the effect of the **learning rate $\eta$** and the number of **epochs** on convergence and model stability.

---

### 8. `LR_SGDRegressor_from_Scratch.ipynb`

* **Dataset:** Diabetes Dataset containing `442` samples and `10` features.
* **Implementation:** Built a custom **Stochastic Gradient Descent (SGD)** regressor from scratch.
* **Benchmark:** Compared the custom implementation against Scikit-learn's `SGDRegressor`.
* **Update Strategy:** Parameters are updated using one observation at a time.
* **Results:**

  * Custom SGD: $R^2 \approx 0.418$
  * Scikit-learn SGD: $R^2 \approx 0.432$

---

### 9. `LR_Mini_Batch_GD_Scratch.ipynb`

* **Dataset:** Diabetes Dataset containing `442` samples and `10` features.
* **Implementation:** Developed a vectorized **Mini-Batch Gradient Descent (MBGD)** implementation from scratch.
* **Comparison:** Compared the custom MBGD implementation against baseline OLS and Scikit-learn's `SGDRegressor`.
* **Configuration:**

  * `batch_size ≈ 7`
  * `learning_rate = 0.01`
  * `epochs = 75`
* **Streaming Learning:** Also explored Scikit-learn's `.partial_fit()` method.
* **Results:**

  * Custom MBGD: $R^2 = 0.4472$
  * Scikit-learn `partial_fit`: $R^2 = 0.4315$

---

### 10. `Polynomial_Regression.ipynb`

* **Dataset:** Synthetic non-linear quadratic dataset:

```text
y = 0.8x² + 0.9x + 2 + noise
```

* **Underfitting Analysis:** Demonstrated that standard Linear Regression struggles to model the non-linear relationship, achieving approximately:

```text
R² ≈ 0.2691
```

* **Polynomial Features:** Applied Scikit-learn's `PolynomialFeatures` to transform the original feature space and allow Linear Regression to model curved relationships.
* **Learning Focus:** Demonstrated the difference between a model's **linear functional form** and the ability to model **non-linear relationships through feature transformation**.

---

### 11. `LR_Ridge_Regularization(L2).ipynb`

This notebook introduces **Ridge Regression and L2 Regularization** and demonstrates its effect on both ordinary and high-degree polynomial regression.

* **Dataset:** Scikit-learn's **Diabetes Dataset**, containing `442` samples and `10` numerical predictive features.

* **Baseline Model:** Trained Scikit-learn's `LinearRegression` using an `80/20` train-test split.

* **Baseline Performance:**

  * **$R^2$ Score:** `0.518811`
  * **RMSE:** `48.7271`

* **Ridge Regression:** Introduced Scikit-learn's `Ridge` estimator with the regularization parameter $\alpha$.

```python
R = Ridge(alpha=0.0001)
```

* **Regularized Performance:**

  * **$R^2$ Score:** `0.518973`
  * **RMSE:** `48.7189`

* **Observation:** A slight improvement was observed after introducing L2 regularization.

#### Polynomial Regression + Ridge Regularization

The notebook further demonstrates the effect of Ridge regularization on a high-degree polynomial model.

* **Synthetic Dataset:** Generated `100` observations using:

```text
x₂ = 0.7x₁² - 2x₁ + 3 + noise
```

* **Polynomial Expansion:** Applied `PolynomialFeatures(degree=16)` to create a highly flexible polynomial model.
* **Ridge Pipeline:** Combined polynomial feature expansion and Ridge Regression using a Scikit-learn `Pipeline`.
* **Regularization Strengths Compared:**

```text
α ∈ {0, 20, 200}
```

* **Visualization:** Plotted the original data points together with predictions produced by different regularization strengths.
* **Key Observation:** Increasing $\alpha$ controls the flexibility of the high-degree polynomial model.
* **Experiment Conclusion:** The notebook notes that approximately **$\alpha = 20$ provides an optimum fit**, while stronger regularization can push the model toward underfitting.

---

### 12. `LR_Ridge_Regression_GradientDescent.ipynb`

This notebook extends the study of Ridge Regression by implementing and comparing **L2-regularized regression using different optimization approaches**.

* **Dataset:** Scikit-learn's **Diabetes Dataset**, containing `442` samples and `10` features.
* **Train-Test Split:** Used an `80/20` split with `random_state=4`.

#### 1. SGDRegressor with L2 Regularization

The notebook first uses Scikit-learn's `SGDRegressor` with an L2 penalty:

```python
SGDRegressor(
    penalty='l2',
    max_iter=500,
    eta0=0.1,
    learning_rate='constant',
    alpha=0.001
)
```

* **$R^2$ Score:** `0.446906`
* **Regularization:** `penalty='l2'`
* **Regularization Strength:** `alpha=0.001`

This demonstrates how **Stochastic Gradient Descent can be combined with L2 regularization** during model optimization.

#### 2. Scikit-learn Ridge Regression

The notebook then trains Scikit-learn's `Ridge` model:

```python
Ridge(
    alpha=0.001,
    max_iter=500,
    solver='sparse_cg'
)
```

* **$R^2$ Score:** `0.462501`
* **Regularization Strength:** `alpha=0.001`

The learned coefficients and intercept are also examined to understand the effect of regularization on the model parameters.

#### 3. Ridge Regression from Scratch using Gradient Descent

Finally, the notebook uses a custom `RidgeGD` implementation:

```python
RidgeGD(
    epochs=500,
    alpha=0.001,
    learning_rate=0.005
)
```

* **Epochs:** `500`
* **Regularization parameter:** `alpha=0.001`
* **Learning rate:** `0.005`
* **$R^2$ Score:** `0.473802`

The learned coefficients and intercept are also displayed.

#### Comparison

| Model          | Regularization | Optimization                | $R^2$ Score |
| -------------- | -------------- | --------------------------- | ----------: |
| `SGDRegressor` | L2             | Stochastic Gradient Descent |  `0.446906` |
| `Ridge`        | L2             | Ridge solver                |  `0.462501` |
| `RidgeGD`      | L2             | Gradient Descent            |  `0.473802` |

This notebook provides a practical comparison between **regularized stochastic optimization, Scikit-learn's Ridge implementation, and a custom Gradient Descent-based Ridge implementation**.

---

## Mathematical Overview

### Simple Linear Regression

```text
y = mx + b
```

Where:

* $m$ = slope/coefficient
* $b$ = intercept
* $x$ = input feature
* $y$ = predicted target

---

### Multiple Linear Regression

```text
y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

---

### Polynomial Regression

```text
y = b₀ + b₁x + b₂x² + ... + bₙxⁿ
```

Polynomial Regression is implemented by transforming the original features into polynomial features and then applying Linear Regression.

---

### Ordinary Least Squares

For Simple Linear Regression:

```text
m = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²

b = ȳ - mx̄
```

---

### Normal Equation

For Multiple Linear Regression:

```text
β = (XᵀX)⁻¹Xᵀy
```

This provides an analytical closed-form solution for the regression coefficients.

---

### Ridge Regression — L2 Regularization

Ridge Regression modifies the ordinary least-squares objective by adding a penalty on the squared magnitude of the coefficients:

```text
Cost = Σ(yᵢ - ŷᵢ)² + α Σwⱼ²
```

Here, $\alpha$ controls the strength of regularization.

As $\alpha$ increases, the model increasingly penalizes large coefficient values, encouraging smaller weights and reducing model complexity.

---

### Ridge Regression with Gradient Descent

Ridge Regression can also be optimized iteratively using Gradient Descent.

The optimization process involves two components:

1. **Prediction error**, which tries to minimize the regression loss.
2. **L2 regularization penalty**, which discourages large coefficient values.

This allows Ridge Regression to be implemented without relying exclusively on Scikit-learn's built-in solver.

---

## Optimization Approaches

### 1. Analytical Optimization

* **Ordinary Least Squares (OLS):** Closed-form solution for Simple Linear Regression.
* **Normal Equation:** Matrix-based closed-form solution for Multiple Linear Regression.

---

### 2. Iterative Optimization

* **Batch Gradient Descent:** Uses the entire training dataset for each parameter update.
* **Stochastic Gradient Descent:** Updates parameters using individual observations.
* **Mini-Batch Gradient Descent:** Updates parameters using small subsets of observations.
* **Ridge Gradient Descent:** Extends Gradient Descent by incorporating an L2 regularization penalty.

These approaches demonstrate how regression models can be optimized numerically rather than relying only on analytical solutions.

---

### 3. Regularization

* **Ridge Regression (L2):** Adds a squared coefficient penalty to the loss function.
* **L2 Regularized SGD:** Combines Stochastic Gradient Descent with an L2 penalty.
* **Ridge + Gradient Descent:** Implements regularized regression through an iterative optimization approach.
* **Purpose:** Controls model complexity, shrinks coefficients, and helps reduce overfitting.

---

## Regression Evaluation Metrics

### Mean Absolute Error — MAE

Measures the average absolute difference between actual and predicted values.

### Mean Squared Error — MSE

Squares prediction errors before averaging them, giving larger errors greater influence.

### Root Mean Squared Error — RMSE

```text
RMSE = √MSE
```

RMSE expresses prediction error in the same units as the target variable.

### $R^2$ Score

Measures how much of the variance in the target variable is explained by the regression model.

### Adjusted $R^2$

Extends $R^2$ by taking the number of predictors into account and penalizing unnecessary features.

---

## Assumptions of Linear Regression

1. **Linearity:** Independent variables have a linear relationship with the target.
2. **No Multicollinearity:** Predictors should not be excessively correlated.
3. **Normality of Residuals:** Residual errors should approximately follow a normal distribution.
4. **Homoscedasticity:** Residual variance should remain approximately constant.
5. **No Autocorrelation:** Residuals should be independent of one another.

---

## Libraries & Tools Used

* **Pandas** — Data manipulation, CSV parsing, and DataFrame operations.
* **NumPy** — Numerical computing, vectorized operations, linear algebra, matrix manipulation, and mathematical calculations.
* **Matplotlib** — 2D scatter plots, regression curves, residual visualizations, and model comparisons.
* **Plotly** — Interactive 3D scatter plots and regression planes.
* **Scikit-learn** — Regression estimators, datasets, feature transformations, model evaluation, preprocessing, pipelines, and optimization utilities.
* **Statsmodels** — Statistical diagnostics and Variance Inflation Factor (VIF) calculations.
* **Jupyter Notebook** — Interactive environment for experimentation, visualization, and documenting the learning process.

---

## Key Learnings & Takeaways

* Implemented **Simple Linear Regression** using Scikit-learn and from scratch.
* Derived and implemented **Ordinary Least Squares** mathematically using NumPy.
* Extended Linear Regression to multiple features using both Scikit-learn and the **Normal Equation**.
* Learned and implemented important **regression evaluation metrics**, including MAE, MSE, RMSE, $R^2$, and Adjusted $R^2$.
* Studied the fundamental **assumptions of Linear Regression** and methods for diagnosing them.
* Implemented **Batch Gradient Descent, Stochastic Gradient Descent, and Mini-Batch Gradient Descent** from scratch.
* Compared custom optimization algorithms against Scikit-learn implementations.
* Used **Polynomial Feature Expansion** to model non-linear relationships.
* Studied the effect of **L2 Regularization** through Ridge Regression.
* Observed how the regularization parameter $\alpha$ influences model complexity and coefficient behavior.
* Applied Ridge Regression to a high-degree polynomial model to study the effect of different regularization strengths.
* Implemented and explored **Ridge Regression using Gradient Descent**.
* Compared **SGDRegressor, Scikit-learn Ridge, and custom Ridge Gradient Descent** implementations.
* Developed a stronger understanding of the relationship between **optimization algorithms, regularization, model complexity, and practical machine learning performance**.

---

## Repository Progression

The notebooks in this repository follow a gradual progression:

**Simple Linear Regression**
↓
**OLS From Scratch**
↓
**Regression Metrics**
↓
**Multiple Linear Regression**
↓
**Normal Equation**
↓
**Linear Regression Assumptions**
↓
**Batch Gradient Descent**
↓
**Stochastic Gradient Descent**
↓
**Mini-Batch Gradient Descent**
↓
**Polynomial Regression**
↓
**Ridge Regression / L2 Regularization**
↓
**Ridge Regression with Gradient Descent**

This progression moves from the mathematical foundations of regression toward **optimization, non-linearity, regularization, and iterative model training**, building a practical understanding of how regression models are constructed, optimized, evaluated, and controlled for overfitting.
