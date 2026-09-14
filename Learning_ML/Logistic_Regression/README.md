# Logistic Regression

This folder contains my learning journey through **Logistic Regression**, starting from the basic idea of linear classification and gradually building towards understanding how Logistic Regression works internally.

Each notebook in this folder focuses on a particular concept that contributes to understanding Logistic Regression, from the **Perceptron and linear decision boundaries** to implementing **Logistic Regression using Gradient Descent from scratch**, and finally extending Logistic Regression to **multi-class classification using Softmax Regression**.

---

## 📓 Notebook 1 — Logistic Regression & Perceptron Trick

### Overview

This notebook is my first step into **classification algorithms**, where I explored how a model can learn to separate data points belonging to two different classes.

I started with a simple **binary classification dataset** containing two features, which made it possible to visualize the data and the decision boundary in two dimensions.

The main focus of this notebook was understanding the **Perceptron Trick** and how a linear classifier can learn a decision boundary by repeatedly updating its weights.

### 🧠 What I Learned

#### 1. Binary Classification

* Difference between regression and classification.
* Understanding binary classification.
* Predicting discrete classes instead of continuous values.

#### 2. Linear Decision Boundary

* How a linear classifier separates two classes.
* Understanding the decision boundary geometrically.
* How weights determine the position and orientation of the boundary.

#### 3. Perceptron

* Working principle of the Perceptron.
* Weighted sum of input features.
* Binary prediction.
* Weight updates based on incorrect predictions.

#### 4. Perceptron Trick

* Understanding how the Perceptron updates its weights.
* Moving the decision boundary to improve classification.
* Learning from incorrect predictions.

#### 5. Step Function

* Using a step function for binary classification.
* Converting a score into Class 0 or Class 1.

#### 6. Implementing the Perceptron From Scratch

* Initializing weights.
* Generating predictions.
* Identifying incorrect predictions.
* Updating weights.
* Observing how the decision boundary changes.

#### 7. Understanding Weights Geometrically

Understanding how weights affect:

* Slope
* Position
* Orientation

of the decision boundary.

#### 8. Visualizing the Learning Process

* Plotting data points and decision boundaries.
* Observing how the boundary changes during training.

#### 9. Perceptron vs Logistic Regression

* Understanding that both can produce a linear decision boundary.
* Understanding the conceptual difference between Perceptron learning and Logistic Regression.

### 🔑 Key Takeaways

* Understanding binary classification.
* Understanding linear decision boundaries.
* Understanding the Perceptron and Perceptron Trick.
* Implementing a classifier from scratch.
* Understanding the role of weights.
* Visualizing the learning process.
* Building the foundation required to understand Logistic Regression.

---

## 📓 Notebook 2 — Perceptron Implementation & Visualizations

### Overview

Building directly upon the concepts from Notebook 1, this notebook focuses on the hands-on implementation of the **Perceptron algorithm**.

I implemented the algorithm using Python to observe how the Perceptron iteratively updates its decision boundary until it successfully classifies a 2D dataset.

### 🧠 What I Learned

#### 1. Hands-on Perceptron Training Loop

* Structuring the iterative training process.
* Processing individual data points.
* Applying the binary step function.
* Updating weights dynamically.

#### 2. Decision Boundary Plotting

Using `matplotlib`, I generated 2D scatter plots along with the decision boundary to observe how the boundary shifts during training.

### 🔑 Key Takeaways

* Turning Perceptron theory into Python code.
* Understanding the training loop.
* Applying the step function.
* Implementing weight updates.
* Visualizing the effect of weight updates.
* Strengthening the foundation for Logistic Regression.

---

## 📓 Notebook 3 — Logistic Regression & Gradient Descent

### Overview

This notebook takes the next step from the **Perceptron** and focuses on understanding how **Logistic Regression can be trained using Gradient Descent**.

I generated a simple **2D binary classification dataset** using Scikit-Learn's `make_classification` and used it to visualize the data and decision boundary.

The notebook then compares **Scikit-Learn's Logistic Regression** with a Logistic Regression model implemented manually using **Gradient Descent**.

### 🧠 What I Learned

#### 1. Creating a Binary Classification Dataset

Using `make_classification` to generate a dataset with:

* 100 samples
* 2 features
* 2 classes
* 1 informative feature
* 1 cluster per class

#### 2. Logistic Regression Using Scikit-Learn

* Training Logistic Regression.
* Extracting coefficients.
* Extracting the intercept.
* Calculating the linear decision boundary.

#### 3. Understanding the Sigmoid Function

Implemented the Sigmoid function from scratch:

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

The Sigmoid function converts the raw linear output into a value between **0 and 1**, forming the basis of probability-based predictions.

#### 4. Adding the Intercept

Learned how the intercept can be incorporated into the feature matrix by adding a column of ones:

```text
X → [1, x₁, x₂, ...]
```

#### 5. Implementing Gradient Descent From Scratch

The custom implementation:

1. Adds the intercept term.
2. Initializes weights.
3. Calculates the linear output.
4. Applies the Sigmoid function.
5. Calculates the prediction error.
6. Calculates the gradient.
7. Updates the weights.
8. Repeats the process for multiple iterations.

#### 6. Understanding the Gradient Descent Update

Understanding the iterative learning process:

```text
Prediction
    ↓
Compare with actual value
    ↓
Calculate gradient
    ↓
Update weights
    ↓
Repeat
```

#### 7. Comparing Scikit-Learn With My Own Implementation

Compared:

* Scikit-Learn Logistic Regression
* Custom Gradient Descent Logistic Regression

by examining their learned parameters and decision boundaries.

#### 8. Visualizing Decision Boundaries

Plotted both decision boundaries on the same dataset to connect:

**Weights → Decision Boundary → Classification**

### 🔑 Key Takeaways

* Creating binary classification datasets.
* Understanding Logistic Regression.
* Understanding the Sigmoid function.
* Extracting coefficients and intercept.
* Understanding how weights determine the decision boundary.
* Understanding Gradient Descent.
* Implementing Logistic Regression from scratch.
* Comparing custom implementations with Scikit-Learn.
* Visualizing decision boundaries.

---

## 📓 Notebook 4 — Logistic Regression & Softmax Regression

### Overview

This notebook extends the concepts of **Logistic Regression from binary classification to multi-class classification**.

Using the **Iris dataset**, I explored how Scikit-Learn's `LogisticRegression` can automatically handle a multi-class classification problem and produce probability estimates for each class.

The notebook also visualizes the learned **multi-class decision regions**.

### 🧠 What I Learned

#### 1. Multi-Class Classification

I moved from binary classification to a problem involving **three different classes** using the Iris dataset:

* Setosa
* Versicolor
* Virginica

This helped me understand how Logistic Regression can be extended beyond two classes.

#### 2. Preparing the Iris Dataset

I used the Iris dataset from Seaborn and:

* Encoded the categorical target using `LabelEncoder`.
* Selected `sepal_length` and `petal_length` as features.
* Used `species` as the target variable.
* Split the data into training and testing sets.

#### 3. Logistic Regression for Multi-Class Classification

I used Scikit-Learn's:

```python
from sklearn.linear_model import LogisticRegression
```

The model automatically recognizes that the problem is **multi-class classification**.

#### 4. Softmax Regression

I learned the idea behind **Softmax Regression**, which extends Logistic Regression to multiple classes.

Instead of producing a probability for only one class, the model produces a probability for **each possible class**.

For example:

```text
Setosa       → 72.5%
Versicolor   → 27.3%
Virginica    → 0.04%
```

The predicted class is then the class with the highest probability.

#### 5. `predict_proba()`

I used:

```python
clf.predict_proba(query)
```

to obtain the probability distribution across all classes for a new input.

This helped me understand the difference between:

```python
clf.predict()
```

and

```python
clf.predict_proba()
```

where `predict()` returns the predicted class while `predict_proba()` provides the probabilities for each class.

#### 6. Confusion Matrix and Accuracy

I evaluated the multi-class classifier using:

* Accuracy Score
* Confusion Matrix

This helped me understand how classification metrics can also be applied to multi-class problems.

#### 7. Decision Region Visualization

Using `mlxtend`, I visualized the decision regions learned by the classifier.

This made it possible to see how the feature space is divided among:

```text
Setosa
Versicolor
Virginica
```

and provided a geometric understanding of multi-class classification.

### 🔑 Key Takeaways

* Understanding the difference between binary and multi-class classification.
* Extending Logistic Regression to multiple classes.
* Understanding the basic idea of Softmax Regression.
* Working with the Iris dataset.
* Encoding categorical target variables.
* Using `predict()` for class predictions.
* Using `predict_proba()` for class probabilities.
* Evaluating multi-class classification using accuracy and confusion matrix.
* Visualizing multi-class decision regions.
* Understanding how Logistic Regression can handle multi-class problems.

---

## 🔗 My Learning Progression

The progression of concepts covered across the notebooks is:

```text
Binary Classification
        ↓
Linear Decision Boundary
        ↓
Step Function
        ↓
Perceptron
        ↓
Perceptron Trick & Weight Updates
        ↓
Hands-on Perceptron Implementation
        ↓
Decision Boundary Visualizations
        ↓
Logistic Regression
        ↓
Sigmoid Function
        ↓
Probability-Based Prediction
        ↓
Gradient Descent
        ↓
Logistic Regression From Scratch
        ↓
Scikit-Learn vs Custom Implementation
        ↓
Decision Boundary Comparison
        ↓
Multi-Class Classification
        ↓
Softmax Regression
        ↓
Class Probability Prediction
        ↓
Confusion Matrix for Multi-Class Problems
        ↓
Multi-Class Decision Regions
```

---

## 🚀 Current Learning Status

Through these notebooks, I have progressed from understanding a basic **linear classifier** to understanding the fundamental mechanics of **Logistic Regression**, and now towards **multi-class classification using Softmax Regression**.

### Concepts Covered So Far

* Binary Classification
* Linear Decision Boundaries
* Perceptron
* Perceptron Trick
* Step Function
* Weight Updates
* Sigmoid Function
* Probability-Based Prediction
* Gradient Descent
* Logistic Regression From Scratch
* Scikit-Learn Logistic Regression
* Multi-Class Classification
* Softmax Regression
* `predict_proba()`
* Confusion Matrix
* Decision Region Visualization

### 📌 Next Concepts to Explore

The next step in this learning journey is to explore the mathematical and optimization foundations of Logistic Regression more deeply, including:

* Logistic Regression Cost / Loss Function
* Log Loss / Binary Cross-Entropy
* Softmax Function Mathematics
* Gradient Calculation
* Optimization
* Learning Rate
* Model Convergence
* Classification Threshold
* Regularization
* Multiclass Loss Functions

This progression is helping me understand not only **how to use ML algorithms**, but also **how they work internally**.
