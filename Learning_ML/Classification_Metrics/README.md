# Classification Metrics

This folder contains my learning and hands-on implementation of **classification evaluation metrics** using **Scikit-Learn**.

The notebooks focus on understanding how to evaluate classification models beyond just accuracy, using **Logistic Regression, Decision Tree, and SVM** on practical datasets.

---

## 📓 Notebook 1: `Classification_Metrics.ipynb`

### Topics

* **Accuracy Score**
* **Confusion Matrix**
* **Precision**
* **Recall**
* **F1 Score**
* **Class-wise Metrics**
* **Macro Average**
* **Weighted Average**
* **Support**
* **Classification Report**
* Comparing **Logistic Regression** and **Decision Tree Classifier**
* Practical evaluation using **Heart Disease** and **Digits** datasets
* Using `sklearn.metrics` for model evaluation

### Learning Outcomes

* Understand how **Accuracy** measures overall correct predictions.
* Read and interpret a **Confusion Matrix**.
* Understand the difference between **Precision, Recall, and F1 Score**.
* Calculate and interpret **class-wise metrics**.
* Understand **Macro** and **Weighted** averaging.
* Understand **Support** and its relation to class frequency.
* Generate and interpret a **Classification Report**.
* Compare classification models using multiple evaluation metrics.
* Understand why **accuracy alone is not sufficient** for evaluating a classification model.

---

## 📓 Notebook 2: `Classification_Metrics_ROC&AUC_Curve.ipynb`

### Topics

* **Probability Scores** using `predict_proba()`
* **Classification Threshold**
* **True Positive Rate (TPR)**
* **False Positive Rate (FPR)**
* **ROC Curve**
* **ROC Curve Visualization**
* Understanding **Thresholds** and their effect on TPR/FPR
* Finding an **Optimal Classification Threshold**
* **AUC — Area Under the Curve**
* **ROC-AUC Score**
* Comparing **Logistic Regression** and **SVM** using ROC-AUC
* **Feature Scaling** using `StandardScaler` for SVM
* Practical evaluation using the **Pima Indians Diabetes Dataset**

### Learning Outcomes

* Understand how classification models produce **probability scores**.
* Understand how changing the **classification threshold** affects predictions.
* Understand **TPR (Sensitivity)** and **FPR**.
* Build and interpret a **ROC Curve**.
* Understand the relationship between **threshold, TPR, and FPR**.
* Find an **optimal threshold** using `TPR - FPR`.
* Understand **AUC** as a measure of model discrimination.
* Calculate and interpret the **ROC-AUC Score**.
* Compare different classification models using **ROC-AUC**.
* Understand the importance of **feature scaling** when working with SVM.

---

## 🧠 Learning Progression

```text
Classification
      ↓
Model Predictions
      ↓
Accuracy
      ↓
Confusion Matrix
      ↓
Precision + Recall
      ↓
F1 Score
      ↓
Class-wise Evaluation
      ↓
Macro & Weighted Average
      ↓
Support
      ↓
Classification Report
      ↓
Probability Scores
      ↓
Classification Threshold
      ↓
TPR + FPR
      ↓
ROC Curve
      ↓
Optimal Threshold
      ↓
AUC / ROC-AUC
      ↓
Model Comparison
```

---

## 🛠️ Libraries Used

* Python
* NumPy
* Pandas
* Matplotlib
* Plotly
* Scikit-Learn

---

## 🎯 Key Takeaway

> **A classification model should not be judged by accuracy alone.**

> Metrics such as **Precision, Recall, F1 Score, ROC-AUC, and the Confusion Matrix** provide a more complete understanding of model performance.

> **ROC Curve and AUC** help evaluate how well a model distinguishes between classes across different classification thresholds.
