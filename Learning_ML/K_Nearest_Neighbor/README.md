# K-Nearest Neighbors (KNN)

This folder contains my implementation and exploration of the **K-Nearest Neighbors (KNN) Classification Algorithm** using Python and Scikit-learn.

I applied KNN to the Breast Cancer dataset to understand how distance-based classification works, how preprocessing affects the model, and how the choice of `K` influences prediction accuracy.

## 📚 Topics Covered & Learned

### 1. Dataset Loading and Exploration
- Imported and explored the Breast Cancer dataset using Pandas.
- Loaded the dataset using `pd.read_csv()`.
- Examined the dataset using `head()` and `shape`.
- Understood the dataset features and target variable.
- Removed unnecessary columns such as `id` and `Unnamed: 32`.

### 2. Data Preprocessing
- Separated independent features (`X`) and the target variable (`y`).
- Used `train_test_split()` to divide the dataset into training and testing sets.
- Applied an 80:20 train-test split.
- Used `random_state=2` for reproducibility.

### 3. Feature Scaling
- Learned why feature scaling is important for distance-based algorithms like KNN.
- Implemented **Standardization** using `StandardScaler`.
- Used `fit_transform()` on training data.
- Used `transform()` on testing data to avoid data leakage.

### 4. K-Nearest Neighbors Classification
- Imported `KNeighborsClassifier` from Scikit-learn.
- Initialized the KNN classifier with `n_neighbors=5`.
- Trained the model using `fit()`.
- Generated predictions using `predict()`.

### 5. Model Evaluation
- Evaluated model performance using `accuracy_score()`.
- Compared predicted labels with actual labels.
- Achieved an accuracy of approximately **97.37%** with K = 5 on the test dataset.

### 6. Hyperparameter Tuning — Finding the Best K
- Experimented with different values of K, ranging from 1 to 15.
- Trained separate KNN models for each value of K.
- Calculated and stored the accuracy for every model.
- Visualized the relationship between K and model accuracy using Matplotlib.
- Identified K = 3 as the best-performing value in this experiment.

### 7. Decision Boundary Visualization
- Used Scikit-learn's built-in Breast Cancer dataset.
- Selected the first two features for two-dimensional visualization.
- Standardized the selected features using `StandardScaler`.
- Implemented a custom `plot_decision_boundaries()` function.
- Used `ListedColormap` to distinguish classification regions.
- Used NumPy's `meshgrid()` to generate coordinate points for plotting.
- Visualized how KNN classifies different regions of the feature space.
- Used `ipywidgets.interact()` to interactively change K from 1 to 20 and observe changes in decision boundaries.

## 🛠️ Technologies & Libraries Used

| Library | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations and meshgrid |
| Matplotlib | Accuracy plots and decision boundaries |
| Scikit-learn | KNN, preprocessing, dataset and evaluation |
| ipywidgets | Interactive visualization |

## 📂 Files

- `KNN.ipynb` — Jupyter Notebook containing data preprocessing, KNN implementation, model evaluation, hyperparameter tuning and decision-boundary visualization.

## 🎯 Key Takeaways

Through this implementation, I learned:

- How KNN classifies data based on the nearest data points.
- Why feature scaling is particularly important for distance-based algorithms.
- How to train and evaluate a classification model using Scikit-learn.
- How changing the value of K affects classification accuracy.
- How to visualize decision boundaries and understand the behaviour of KNN.
- How to perform basic hyperparameter experimentation.

---

**Learning Journey:** Exploring Machine Learning algorithms through practical implementations and experiments.