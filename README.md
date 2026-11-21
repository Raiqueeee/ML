# ML

A collection of machine learning projects.

This repository brings together the machine learning projects I build, in one place. Each project is a complete, runnable notebook covering the full workflow: loading data, exploring it, preparing it, training a model, and evaluating the results.

## Projects

### 1. Random Forest Classifier on the Iris Dataset

Predict the species of an iris flower from its measurements.

- **Notebook:** `Random_Forest_Classifier.ipynb`
- **Dataset:** Iris dataset, built into scikit-learn (`sklearn.datasets.load_iris`) — 150 samples, 4 features (sepal length, sepal width, petal length, petal width), 3 classes: setosa, versicolor, virginica (50 samples each), no missing values
- **Algorithm:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`) with 100 trees (`n_estimators=100`, `random_state=42`)
- **Split:** 80/20 train/test split (`test_size=0.20`, `random_state=42`, `stratify=y`) — 120 samples for training, 30 for testing
- **Results:** Accuracy 90% (0.9000) — 27 of 30 test samples classified correctly

Confusion matrix:

```
[[10,  0,  0],
 [ 0,  9,  1],
 [ 0,  2,  8]]
```

| Class | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| setosa | 1.00 | 1.00 | 1.00 |
| versicolor | 0.82 | 0.90 | 0.86 |
| virginica | 0.89 | 0.80 | 0.84 |

Feature importances: petal width (cm) 0.4372, petal length (cm) 0.4315, sepal length (cm) 0.1163, sepal width (cm) 0.0150. The notebook also shows the confusion matrix as a heatmap and the feature importances as a bar chart.

### 2. XGBoost on the Breast Cancer Dataset

Predict whether a breast tumour is malignant or benign from cell nucleus measurements.

- **Notebook:** `XGBoost.ipynb`
- **Dataset:** Breast Cancer Wisconsin dataset, built into scikit-learn (`sklearn.datasets.load_breast_cancer`) — 569 samples, 30 features computed from cell nucleus measurements, and a binary target: malignant (0) or benign (1); 212 malignant and 357 benign, no missing values
- **Algorithm:** XGBoost Classifier (`xgboost.XGBClassifier`) with 100 trees (`n_estimators=100`, `random_state=42`, `eval_metric='logloss'`)
- **Split:** 80/20 train/test split (`test_size=0.20`, `random_state=42`, `stratify=y`) — 455 samples for training, 114 for testing
- **Results:** Accuracy 95.61% (0.9561) — 109 of 114 test samples classified correctly

Confusion matrix:

```
[[38,  4],
 [ 1, 71]]
```

| Class | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| malignant | 0.97 | 0.90 | 0.94 |
| benign | 0.95 | 0.99 | 0.97 |

The notebook also shows the confusion matrix as a heatmap and the feature importances as a bar chart.

### 3. Linear Regression on the Diabetes Dataset

Predict disease progression in diabetes patients from their measurements.

- **Notebook:** `Linear_Regression.ipynb`
- **Dataset:** Diabetes dataset, built into scikit-learn (`sklearn.datasets.load_diabetes`), 442 samples, 10 features (age, sex, body mass index, average blood pressure, and six blood serum measurements)
- **Algorithm:** Linear Regression (`sklearn.linear_model.LinearRegression`)
- **Split:** 80/20 train/test split (`test_size=0.20`, `random_state=42`) — 353 samples for training, 89 for testing
- **Results:** R-squared 0.4526 — mean squared error 2900.19, root mean squared error 53.85, mean absolute error 42.79
The notebook also shows actual vs predicted values and the residuals as charts.

### 4. Logistic Regression on the Breast Cancer Dataset

Predict whether a breast tumour is malignant or benign from cell nucleus measurements.

- **Notebook:** `Logistic_Regression.ipynb`
- **Dataset:** Breast Cancer Wisconsin dataset, built into scikit-learn (`sklearn.datasets.load_breast_cancer`), 569 samples, 30 features (cell nucleus measurements)
- **Algorithm:** Logistic Regression (`sklearn.linear_model.LogisticRegression`) with scaling (`sklearn.preprocessing.StandardScaler`, `max_iter=1000`, `random_state=42`)
- **Split:** 80/20 train/test split (`test_size=0.20`, `random_state=42`, `stratify=y`) — 455 samples for training, 114 for testing
- **Results:** Accuracy 98.25% (0.9825) — 112 of 114 test samples classified correctly
Confusion matrix:
```
[[41,  1],
 [ 1, 71]]
```

| Class | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| malignant | 0.98 | 0.98 | 0.98 |
| benign | 0.99 | 0.99 | 0.99 |

The notebook also shows the confusion matrix as a heatmap and the top 10 features by coefficient as a bar chart.
### 5. Decision Tree Classifier on the Iris Dataset

Predict the species of an iris flower from its measurements using a shallow, easy-to-read decision tree.

- **Notebook:** `Decision_Tree_Classifier.ipynb`
- **Dataset:** Iris dataset, built into scikit-learn (`sklearn.datasets.load_iris`), 150 samples, 4 features (sepal length, sepal width, petal length, petal width)
- **Algorithm:** Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`) with a maximum depth of 3 (`max_depth=3`, `random_state=42`)
- **Split:** 80/20 train/test split (`test_size=0.20`, `random_state=42`, `stratify=y`) — 120 samples for training, 30 for testing
- **Results:** Accuracy 96.67% (0.9667) — 29 of 30 test samples classified correctly
Confusion matrix:
```
[[10, 0, 0],
 [ 0, 9, 1],
 [ 0, 0, 10]]
```

| Class | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| setosa | 1.00 | 1.00 | 1.00 |
| versicolor | 1.00 | 0.90 | 0.95 |
| virginica | 0.91 | 1.00 | 0.95 |
The notebook also shows the confusion matrix as a heatmap, the feature importances as a bar chart, and the decision tree itself as a plot.


### 6. K-Nearest Neighbors on the Iris Dataset

Predict the species of an iris flower from its measurements by looking at the 5 nearest training samples.

- **Notebook:** `K_Nearest_Neighbors.ipynb`
- **Dataset:** Iris dataset, built into scikit-learn (`sklearn.datasets.load_iris`), 150 samples, 4 features (sepal length, sepal width, petal length, petal width)
- **Algorithm:** K-Nearest Neighbors (`sklearn.neighbors.KNeighborsClassifier`) with k=5 neighbours (`n_neighbors=5`), with feature scaling (`sklearn.preprocessing.StandardScaler`)
- **Split:** 80/20 train/test split (`test_size=0.20`, `random_state=42`, `stratify=y`) — 120 samples for training, 30 for testing
- **Results:** Accuracy 93.33% (0.9333) — 28 of 30 test samples classified correctly
Confusion matrix:
```
[[10, 0, 0],
 [ 0, 10, 0],
 [ 0, 2, 8]]
```

| Class | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| setosa | 1.00 | 1.00 | 1.00 |
| versicolor | 0.83 | 1.00 | 0.91 |
| virginica | 1.00 | 0.80 | 0.89 |

The notebook also shows the confusion matrix as a heatmap and accuracy for different values of k as a line chart.
### 7. Support Vector Machine on the Breast Cancer Dataset

Predict whether a breast tumour is malignant or benign from cell nucleus measurements using an SVM with the RBF kernel.

- **Notebook:** `Support_Vector_Machine.ipynb`
- **Dataset:** Breast Cancer Wisconsin dataset, built into scikit-learn (`sklearn.datasets.load_breast_cancer`), 569 samples, 30 features (cell nucleus measurements)
- **Algorithm:** Support Vector Machine (`sklearn.svm.SVC`) with the RBF kernel (`kernel='rbf'`, `random_state=42`), with feature scaling (`sklearn.preprocessing.StandardScaler`)
- **Split:** 80/20 train/test split (`test_size=0.20`, `random_state=42`, `stratify=y`) — 455 samples for training, 114 for testing
- **Results:** Accuracy 98.25% (0.9825) — 112 of 114 test samples classified correctly
Confusion matrix:
```
[[41, 1],
 [ 1, 71]]
```
| Class | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| malignant | 0.98 | 0.98 | 0.98 |
| benign | 0.99 | 0.99 | 0.99 |

The notebook also shows the confusion matrix as a heatmap. The model uses 97 support vectors in total: 51 for the malignant class and 46 for the benign class.
### More projects coming

New projects will be added here as they are completed, each with its own notebook and a short entry in the list above.

## How to Run

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Open the notebook you want in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run all cells in order.

## Files

- `Random_Forest_Classifier.ipynb` — Random Forest Classifier on the Iris dataset: imports, data loading, exploration, train/test split, model training, evaluation, and charts
- `XGBoost.ipynb` — XGBoost on the Breast Cancer dataset: imports, data loading, exploration, scaling, train/test split, model training, evaluation, and charts
- `Linear_Regression.ipynb` — Linear Regression on the Diabetes dataset: imports, data loading, exploration, train/test split, model training, evaluation, and charts
- `Logistic_Regression.ipynb` — Logistic Regression on the Breast Cancer dataset: imports, data loading, exploration, scaling, train/test split, model training, evaluation, and charts
- `Decision_Tree_Classifier.ipynb` — Decision Tree Classifier on the Iris dataset: imports, data loading, exploration, train/test split, model training, evaluation, and charts
- `K_Nearest_Neighbors.ipynb` — K-Nearest Neighbors on the Iris dataset: imports, data loading, exploration, scaling, train/test split, model training, evaluation, and charts
- `Support_Vector_Machine.ipynb` — Support Vector Machine on the Breast Cancer dataset: imports, data loading, exploration, scaling, train/test split, model training, evaluation, and charts
- `README.md` — this collection description
- `requirements.txt` — Python dependencies
