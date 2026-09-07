# Logistic Regression

This folder contains my learning journey through **Logistic Regression**, starting from the basic idea of linear classification and gradually building towards understanding Logistic Regression from its fundamentals.

Each notebook in this folder focuses on a particular concept that contributes to understanding how Logistic Regression works.

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

- Regression predicts continuous numerical values.
- Classification predicts discrete classes.

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

- Positive score → Class 1
- Non-positive score → Class 0

This provided an intuitive understanding of how the Perceptron makes classification decisions.

---

### 6. Implementing the Perceptron From Scratch

Instead of directly using a machine-learning library, I implemented the Perceptron learning process myself.

Through this, I understood:

- How weights are initialized.
- How predictions are generated.
- How incorrect predictions are identified.
- How weights are updated.
- How the decision boundary changes as the model learns.

This helped me move away from treating machine-learning algorithms as black boxes.

---

### 7. Understanding Weights Geometrically

I learned that the weights are not just numerical parameters.

They determine the **position and orientation of the decision boundary**.

By changing the weights, the boundary changes its:

- Slope
- Position
- Orientation

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

- What binary classification means.
- How a linear decision boundary separates classes.
- How a Perceptron works.
- How the step function is used for binary prediction.
- How the Perceptron updates its weights.
- How the Perceptron Trick works.
- How weights influence the decision boundary.
- How to implement a basic classifier from scratch.
- How to visualize a classifier's decision boundary.
- How the decision boundary changes during learning.
- The conceptual difference between the Perceptron and Logistic Regression.
- Why understanding the Perceptron is useful before studying Logistic Regression in detail.

---

## 🔗 My Learning Progression

This notebook establishes the foundation for the Logistic Regression concepts that I will explore in the upcoming notebooks.

```text
Binary Classification
        ↓
Linear Decision Boundary
        ↓
Step Function
        ↓
Perceptron
        ↓
Perceptron Trick
        ↓
Weight Updates
        ↓
Decision Boundary Visualization
        ↓
Logistic Regression
```

---

## 🚀 What's Next?

This is the **first notebook** in my Logistic Regression learning path.

Future notebooks will build upon these concepts and explore the mathematical and practical foundations of Logistic Regression in greater depth.

The folder will be updated as I continue learning and implementing new concepts.