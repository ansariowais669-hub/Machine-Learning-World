# Logistic Regression

This folder contains my learning journey through **Logistic Regression**, starting from the basic idea of linear classification and gradually building towards understanding how Logistic Regression works internally.

Each notebook in this folder focuses on a particular concept that contributes to understanding Logistic Regression, from the **Perceptron and linear decision boundaries** to implementing **Logistic Regression using Gradient Descent from scratch**.

---

## 📓 Notebook 1 — Logistic Regression & Perceptron Trick

### Overview

This notebook is my first step into **classification algorithms**, where I explored how a model can learn to separate data points belonging to two different classes.

I started with a simple **binary classification dataset** containing two features, which made it possible to visualize the data and the decision boundary in two dimensions.

The main focus of this notebook was understanding the **Perceptron Trick** and how a linear classifier can learn a decision boundary by repeatedly updating its weights.

---

## 🧠 What I Learned

### 1. Binary Classification

I learned the basic idea behind binary classification, where the goal is to assign each data point to one of two classes.

I also understood how classification differs from regression:

* Regression predicts continuous numerical values.
* Classification predicts discrete classes.

---

### 2. Linear Decision Boundary

I learned how a linear classifier separates two classes using a **decision boundary**.

For a two-feature dataset, this boundary can be represented as a straight line.

The position and orientation of this line depend on the model's learned weights.

---

### 3. Perceptron

I learned the basic working principle of the **Perceptron**, one of the simplest linear classification algorithms.

The Perceptron:

1. Takes the input features.
2. Calculates a weighted sum.
3. Makes a binary prediction.
4. Compares the prediction with the actual class.
5. Updates its weights when the prediction is incorrect.

This helped me understand how a machine-learning model can gradually learn from its mistakes.

---

### 4. Perceptron Trick

The main concept explored in this notebook was the **Perceptron Trick**.

I learned how the model updates its weights based on the difference between the actual and predicted class.

The basic idea is:

> If the model makes a wrong prediction, adjust the weights so that the decision boundary moves in a direction that improves the classification.

Repeated updates gradually move the decision boundary toward a position that correctly separates the classes.

---

### 5. Step Function

I learned how a **step function** can be used to convert the calculated score into a binary prediction.

Conceptually:

* Positive score → Class 1
* Non-positive score → Class 0

This provided an intuitive understanding of how the Perceptron makes classification decisions.

---

### 6. Implementing the Perceptron From Scratch

Instead of directly using a machine-learning library, I implemented the Perceptron learning process myself.

Through this, I understood:

* How weights are initialized.
* How predictions are generated.
* How incorrect predictions are identified.
* How weights are updated.
* How the decision boundary changes as the model learns.

This helped me move away from treating machine-learning algorithms as black boxes.

---

### 7. Understanding Weights Geometrically

I learned that the weights are not just numerical parameters.

They determine the **position and orientation of the decision boundary**.

By changing the weights, the boundary changes its:

* Slope
* Position
* Orientation

This helped me connect the mathematical representation of a classifier with its geometric interpretation.

---

### 8. Visualizing the Learning Process

I visualized the decision boundary along with the data points to understand how the classifier separates the two classes.

I also explored how the decision boundary changes during the learning process.

This made it easier to understand that the model does not magically know the correct boundary—it gradually adjusts its parameters through repeated updates.

---

### 9. Perceptron vs Logistic Regression

After implementing the Perceptron approach, I compared its decision boundary with the one obtained using **Scikit-Learn's Logistic Regression**.

This helped me understand that although both methods can produce a linear decision boundary, they learn that boundary using different approaches.

The Perceptron focuses on correctly classifying the training points, whereas Logistic Regression learns its parameters through an optimization-based approach.

---

## 🔑 Key Takeaways

From this notebook, I learned:

* What binary classification means.
* How a linear decision boundary separates classes.
* How a Perceptron works.
* How the step function is used for binary prediction.
* How the Perceptron updates its weights.
* How the Perceptron Trick works.
* How weights influence the decision boundary.
* How to implement a basic classifier from scratch.
* How to visualize a classifier's decision boundary.
* How the decision boundary changes during learning.
* The conceptual difference between the Perceptron and Logistic Regression.
* Why understanding the Perceptron is useful before studying Logistic Regression in detail.

---

## 📓 Notebook 2 — Perceptron Implementation & Visualizations

### Overview

Building directly upon the concepts from Notebook 1, this notebook focuses on hands-on implementation of the **Perceptron algorithm**.

I implemented the algorithm using Python to observe how the Perceptron iteratively updates its decision boundary until it successfully classifies a 2D dataset.

---

## 🧠 What I Learned

### 1. Hands-on Perceptron Training Loop

I learned how to structure the iterative training loop that goes through individual data points, evaluates the binary step output, and applies the weight update rule dynamically.

---

### 2. Decision Boundary Plotting

Using `matplotlib`, I generated 2D scatter plots of the data along with the decision boundary.

This made it possible to visually observe how the decision boundary shifts as the model updates its weights.

---

## 🔑 Key Takeaways

* Practical experience turning the Perceptron theory into Python code.
* Understanding the training loop of a linear classifier.
* Applying the step function for binary classification.
* Implementing weight updates programmatically.
* Visualizing how weight updates affect the decision boundary.
* Building a stronger foundation for understanding Logistic Regression.

---

## 📓 Notebook 3 — Logistic Regression & Gradient Descent

### Overview

This notebook takes the next step from the **Perceptron** and focuses on understanding how **Logistic Regression can be trained using Gradient Descent**.

I first generated a simple **2D binary classification dataset** using Scikit-Learn's `make_classification`, which allowed me to visualize the data and work with a linear decision boundary.

The notebook then compares the decision boundary obtained from **Scikit-Learn's Logistic Regression** with a Logistic Regression model implemented manually using **Gradient Descent**.

---

## 🧠 What I Learned

### 1. Creating a Binary Classification Dataset

I used Scikit-Learn's `make_classification` to generate a binary classification dataset with:

* 100 samples
* 2 features
* 2 classes
* 1 informative feature
* 1 cluster per class

The dataset was intentionally created so that the classes could be visualized and separated using a linear decision boundary.

---

### 2. Logistic Regression Using Scikit-Learn

I implemented Logistic Regression using:

```python
from sklearn.linear_model import LogisticRegression
```

I trained the model on the generated dataset and examined its:

* Coefficients
* Intercept

I also used the learned coefficients and intercept to calculate the equation of the model's **linear decision boundary**.

---

### 3. Understanding the Sigmoid Function

I implemented the **Sigmoid function** manually:

```python
def sigmoid(z):
    return 1/(1 + np.exp(-z))
```

The sigmoid function converts the model's raw linear output into a value between **0 and 1**, which forms the basis for Logistic Regression's probability-based predictions.

---

### 4. Adding the Intercept

I learned how the intercept can be incorporated into the input matrix by adding a column of ones.

Conceptually:

```text
X → [1, x₁, x₂, ...]
```

This allows the intercept to be treated as an additional weight during the Gradient Descent process.

---

### 5. Implementing Gradient Descent From Scratch

The main focus of this notebook was creating a custom Gradient Descent implementation rather than relying entirely on Scikit-Learn.

The implementation:

1. Adds the intercept term to the feature matrix.
2. Initializes the weights.
3. Calculates the linear output.
4. Applies the sigmoid function.
5. Compares the predicted probabilities with the actual values.
6. Calculates the gradient.
7. Updates the weights.
8. Repeats the process for multiple iterations.

The custom implementation performs **5000 iterations** with a learning rate of **0.5**.

---

### 6. Understanding the Gradient Descent Update

I learned how the weights can be updated using the gradient obtained from the difference between the actual values and the predicted probabilities.

The implementation follows the idea:

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

This helped me understand how Logistic Regression can learn its parameters iteratively instead of treating the model as a black box.

---

### 7. Comparing Scikit-Learn With My Own Implementation

After training Logistic Regression using Scikit-Learn, I trained another model using my own Gradient Descent implementation.

I extracted the coefficients and intercept from both approaches and calculated their respective decision boundaries.

This allowed me to visually compare:

* Scikit-Learn Logistic Regression
* Custom Gradient Descent Logistic Regression

The comparison helped reinforce the idea that a library implementation is ultimately performing the same fundamental learning process using an optimization procedure.

---

### 8. Visualizing the Decision Boundaries

I plotted both decision boundaries on the same 2D dataset.

This provided a visual way to compare the model trained using Scikit-Learn with the model trained using my own Gradient Descent implementation.

The visualization helped connect:

**Weights → Decision Boundary → Classification**

and made the mathematical concepts easier to understand.

---

## 🔑 Key Takeaways

From this notebook, I learned:

* How to create a binary classification dataset using `make_classification`.
* How Logistic Regression is implemented using Scikit-Learn.
* How to extract Logistic Regression coefficients and intercept.
* How the coefficients determine the linear decision boundary.
* What the Sigmoid function does.
* How to implement the Sigmoid function from scratch.
* How the intercept can be incorporated into the feature matrix.
* The basic idea behind Gradient Descent.
* How gradients are calculated from predictions and actual values.
* How weights are updated iteratively.
* How to implement Logistic Regression training using Gradient Descent.
* How to compare a custom implementation with Scikit-Learn.
* How to visualize and compare decision boundaries.
* How Logistic Regression builds upon the concepts learned from the Perceptron.

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
```

---

## 🚀 Current Learning Status

Through these notebooks, I have progressed from understanding a basic **linear classifier** to understanding the fundamental mechanics behind **Logistic Regression**.

The next step in this learning journey is to explore the mathematical foundation behind Logistic Regression more deeply, including concepts such as:

* Logistic Regression cost/loss function
* Log Loss / Binary Cross-Entropy
* Gradient calculation
* Optimization
* Learning rate
* Model convergence
* Probability and classification threshold
* Regularization

This progression is helping me understand not only **how to use ML algorithms**, but also **how they work internally**.
