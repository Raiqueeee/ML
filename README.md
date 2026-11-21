# Random Forest Classifier on the Iris Dataset

## Overview

This project trains a Random Forest Classifier to predict the species of an iris flower from its measurements. It is a beginner-friendly classification project that covers the full workflow: loading data, exploring it, splitting it into training and test sets, training a model, and evaluating the results.

## Dataset

The Iris dataset is built into scikit-learn (`sklearn.datasets.load_iris`).

- 150 samples
- 4 features: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
- 3 classes: setosa, versicolor, virginica (50 samples each)
- No missing values

## Algorithm

Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`) with 100 trees (`n_estimators=100`, `random_state=42`).

Tree-based models do not require feature scaling, because splits are based on feature thresholds rather than distances, so no scaling was applied. The data was split into training and test sets with an 80/20 split (`test_size=0.20`, `random_state=42`, `stratify=y`): 120 samples for training and 30 for testing, with 10 samples of each class in the test set.

## How to Run

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Open `Random_Forest_Classifier.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run all cells in order.

## Results

- **Accuracy: 90% (0.9000) — 27 of 30 test samples classified correctly**

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

Feature importances: petal width (cm) 0.4372, petal length (cm) 0.4315, sepal length (cm) 0.1163, sepal width (cm) 0.0150. Petal measurements are the most useful features for telling the species apart.

The notebook also shows the confusion matrix as a heatmap and the feature importances as a bar chart.

## Files

- `Random_Forest_Classifier.ipynb` — the complete notebook: imports, data loading, exploration, train/test split, model training, evaluation, and charts
- `README.md` — this project description
- `requirements.txt` — Python dependencies
