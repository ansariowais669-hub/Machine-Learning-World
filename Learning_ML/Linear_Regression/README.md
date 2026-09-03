# Linear, Polynomial & Regularized Regression

This repository contains my practice and hands-on implementations of **regression algorithms using Python**, with an emphasis on understanding the mathematical foundations behind each model and implementing them both through **Scikit-learn** and **from scratch** where applicable.

The repository progressively explores **Simple Linear Regression, Multiple Linear Regression, Regression Evaluation Metrics, Linear Regression Assumptions, Batch Gradient Descent, Stochastic Gradient Descent, Mini-Batch Gradient Descent, Polynomial Regression, and Ridge Regression (L2 Regularization)**.

The primary goal of this repository is to bridge **theoretical mathematical concepts with practical machine learning implementations**, including model training, optimization, evaluation, diagnostics, regularization, visualization, and comparison between different approaches.

---

## Notebooks

### 1. `LR_Simple_LinearRegression.ipynb`

* **Dataset:** Placement dataset containing **CGPA vs. Package**.

* **Concepts:** Exploratory Data Analysis using Pandas and splitting the dataset using `train_test_split`.

* **Implementation:** Built a Simple Linear Regression model using Scikit-learn's `LinearRegression`.

* **Visualization:** Plotted the fitted 2D regression line together with the actual observations.

* **Mathematical Foundation:** Studied the relationship between the learned slope, intercept, and the linear equation:

  \(y = mx + b\)

---

### 2. `LR_SimpleLinearRegression_OLS_scratch.ipynb`

* **Concepts:** Implemented Simple Linear Regression completely from scratch without using a machine learning estimator.

* **Implementation:** Created a custom `LR` class with object-oriented `fit()` and `predict()` methods.

* **Mathematical Foundation:** Implemented the closed-form **Ordinary Least Squares (OLS)** equations using NumPy:

  \(m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}\)

  \(b = \bar{y} - m\bar{x}\)

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

* **Adjusted $R^2$:** Demonstrated how Adjusted $R^2$ penalizes the addition of non-informative features:

  $$$\text{Adjusted } R^2 =
  1 -
  \left[
  \frac{(1-R^2)(n-1)}
  {n-k-1}
  \right]$$
  $$$

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

  \(\beta = (X^T X)^{-1}X^Ty\)

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

* **Gradient Equations:**

  $$$\frac{\partial L}{\partial b}
  =
  -2\sum(y_i-(mx_i+b))$$

  $$\frac{\partial L}{\partial m}
  =
  -2\sum(y_i-(mx_i+b))x_i$$

  $$$

* **Hyperparameter Analysis:** Studied the effect of the **learning rate $\eta$** and the number of **epochs** on convergence and model stability.

---

### 8. `LR_SGDRegressor_from_Scratch.ipynb`

* **Dataset:** Diabetes Dataset containing `442` samples and `10` features.

* **Implementation:** Built a custom **Stochastic Gradient Descent (SGD)** regressor from scratch.

* **Benchmark:** Compared the custom implementation against Scikit-learn's `SGDRegressor`.

* **Update Strategy:** Parameters are updated using one observation at a time:

  \(\hat{y} = X_i \cdot W + b\)

  $$$\frac{\partial L}{\partial b}
  =
  -2(y_i-\hat{y})$$

  $$\frac{\partial L}{\partial W}
  =
  -2(y_i-\hat{y})X_i$$

  $$$

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

* **Gradient Calculation:**

  $$$\frac{\partial L}{\partial W}
  =
  -2(y_{\text{batch}}-\hat{y})^T X_{\text{batch}}$$

  $$$

* **Streaming Learning:** Also explored Scikit-learn's `.partial_fit()` method.

* **Results:**

  * Custom MBGD: $R^2 = 0.4472$
  * Scikit-learn `partial_fit`: $R^2 = 0.4315$

---

### 10. `Polynomial_Regression.ipynb`

* **Dataset:** Synthetic non-linear quadratic dataset:

  \(y = 0.8x^2 + 0.9x + 2 + \text{noise}\)

* **Underfitting Analysis:** Demonstrated that standard Linear Regression struggles to model the non-linear relationship, achieving approximately:

  \(R^2 \approx 0.2691\)

* **Polynomial Features:** Applied Scikit-learn's `PolynomialFeatures` to transform the original feature space and allow Linear Regression to model curved relationships.

* **Learning Focus:** Demonstrated the difference between a model's **linear functional form** and the ability to model **non-linear relationships through feature transformation**.

---

### 11. `LR_Ridge_Regularization(L2).ipynb`

* **Dataset:** Scikit-learn's **Diabetes Dataset**, containing `442` samples and `10` numerical predictive features.

* **Baseline Model:** Trained Scikit-learn's `LinearRegression` model after splitting the dataset into training and testing sets using an `80/20` split.

* **Baseline Performance:**

  * **$R^2$ Score:** `0.518811`
  * **RMSE:** `48.7271`

* **Ridge Regression:** Introduced Scikit-learn's `Ridge` estimator with the regularization parameter $\alpha$:

  ```python
  R = Ridge(alpha=0.0001)
  ```

* **Regularized Performance:**

  * **$R^2$ Score:** `0.518973`
  * **RMSE:** `48.7189`

* **Observation:** The notebook demonstrates a **slight improvement** after introducing L2 regularization.

#### Polynomial Regression + Ridge Regularization

The notebook further demonstrates the effect of Ridge regularization on a high-degree polynomial model.

* **Synthetic Dataset:** Generated `100` observations using:

  \(x_2 = 0.7x_1^2 - 2x_1 + 3 + \text{noise}\)

* **Polynomial Expansion:** Applied `PolynomialFeatures(degree=16)` to deliberately create a highly flexible polynomial model.

* **Ridge Pipeline:** Combined polynomial feature expansion and Ridge Regression using a Scikit-learn `Pipeline`.

* **Regularization Strengths Compared:**

  \(\alpha \in \{0,20,200\}\)

* **Visualization:** Plotted the original data points together with the predictions produced by the different regularization strengths.

* **Key Observation:** The notebook demonstrates how increasing the regularization parameter controls the flexibility of the high-degree polynomial model.

* **Experiment Conclusion:** The notebook notes that approximately **$\alpha = 20$ provides an optimum fit**, while stronger regularization such as **$\alpha = 100$ leads toward underfitting**.

---

## Mathematical Overview

### Simple Linear Regression

\(y = mx+b\)

Where:

* $m$ = slope/coefficient
* $b$ = intercept
* $x$ = input feature
* $y$ = predicted target

---

### Multiple Linear Regression

\(y = b_0+b_1x_1+b_2x_2+\dots+b_nx_n\)

---

### Polynomial Regression

\(y=b_0+b_1x+b_2x^2+\dots+b_nx^n\)

Polynomial Regression is implemented by transforming the original features into polynomial features and then applying Linear Regression.

---

### Ordinary Least Squares

For Simple Linear Regression:

$$$m =
\frac{
\sum(x_i-\bar{x})(y_i-\bar{y})
}{
\sum(x_i-\bar{x})^2
}$$

$$b=\bar{y}-m\bar{x}$$

---

### Normal Equation

For Multiple Linear Regression:

$$\beta=(X^TX)^{-1}X^Ty$$

This provides an analytical closed-form solution for the regression coefficients.

---

### Ridge Regression — L2 Regularization

Ridge Regression modifies the ordinary least-squares objective by adding a penalty on the squared magnitude of the coefficients:

$$$

# \text{Cost}

\sum_{i=1}^{n}(y_i-\hat{y}*i)^2
+
\alpha\sum*{j=1}^{p}w_j^2

$$$

Here, $\alpha$ controls the strength of regularization.

As $\alpha$ increases, the model increasingly penalizes large coefficient values, encouraging smaller weights and reducing model complexity.

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

These approaches demonstrate how regression models can be optimized numerically rather than relying only on analytical solutions.

---

### 3. Regularization

* **Ridge Regression (L2):** Adds a squared coefficient penalty to the loss function.

* **Purpose:** Controls model complexity, shrinks coefficients, and helps reduce overfitting.

---

## Regression Evaluation Metrics

### Mean Absolute Error — MAE

Measures the average absolute difference between actual and predicted values.

### Mean Squared Error — MSE

Squares prediction errors before averaging them, giving larger errors greater influence.

### Root Mean Squared Error — RMSE

$$RMSE=\sqrt{MSE}$$

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

---

## Key Learnings & Takeaways

* Implemented **Simple Linear Regression** using Scikit-learn and from scratch.

* Derived and implemented **Ordinary Least Squares** mathematically using NumPy.

* Extended linear regression to multiple features using both Scikit-learn and the **Normal Equation**.

* Learned and implemented important **regression evaluation metrics**, including MAE, MSE, RMSE, $R^2$, and Adjusted $R^2$.

* Studied the fundamental **assumptions of Linear Regression** and methods for diagnosing them.

* Implemented **Batch Gradient Descent, Stochastic Gradient Descent, and Mini-Batch Gradient Descent** from scratch.

* Compared custom optimization algorithms against Scikit-learn implementations.

* Used **Polynomial Feature Expansion** to model non-linear relationships.

* Studied the effect of **L2 Regularization** through Ridge Regression.

* Observed how the regularization parameter $\alpha$ influences model complexity and coefficient behavior.

* Applied Ridge Regression to a high-degree polynomial model to visualize the transition between **overfitting, optimal fitting, and underfitting**.

* Built a stronger understanding of the relationship between **mathematical optimization, model complexity, regularization, and practical machine learning performance**.

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

This progression moves from the mathematical foundations of regression toward **optimization, non-linearity, and regularization**, building a practical understanding of how regression models are constructed, optimized, evaluated, and controlled for overfitting.
$$$
