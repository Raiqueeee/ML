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
- `README.md` — this collection description
- `requirements.txt` — Python dependencies
