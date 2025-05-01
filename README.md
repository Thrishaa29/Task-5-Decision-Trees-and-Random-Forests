# Task-5-Decision-Trees-and-Random-Forests
This repository demonstrates a complete workflow for building, tuning, and evaluating tree‑based machine learning models on the Heart Disease dataset.

Contents

heart.csv: Raw dataset (ensure it’s in the project root).

train_model.py: Python script that implements all steps from data loading to visualization.

README.md: This file, detailing the processes and commands.

1. Data Loading & Preparation

Load data from heart.csv using pandas.read_csv.

Separate features (X) and label (y = target).

No missing-value imputation or encoding is needed (all features are numeric).

import pandas as pd
df = pd.read_csv('heart.csv')
X = df.drop('target', axis=1)
y = df['target']

2. Train/Test Split
Use sklearn.model_selection.train_test_split with test_size=0.3 and random_state=42.
Result: 70% training data, 30% held-out test data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =
    train_test_split(X, y, test_size=0.3, random_state=42)

3. Decision Tree Classifier
Full-depth tree (no constraints) to measure overfitting:
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

2. **Shallow tree** (limit `max_depth=3`) to control complexity:
   ```python
dt_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_shallow.fit(X_train, y_train)
Evaluate both with .score() on train and test sets.

4. Overfitting Analysis
Compare train vs. test accuracy: 100% vs. ~97% for full tree indicates overfitting.
Depth‑3 tree yields closer train/test scores (~86% vs. ~80%), improving generalization.

5. Random Forest Classifier
Train a default RandomForestClassifier(random_state=42) on the same split.
Evaluate train/test accuracy to assess ensemble performance.
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

6. Feature Importance
Extract .feature_importances_ from the trained Random Forest.
Plot importances with Matplotlib’s horizontal bar chart.
importances = rf.feature_importances_
# Sort and plot...

7. Cross-validation
Use cross_val_score (5-fold) on the entire dataset:
Shallow Decision Tree (depth=3)
Random Forest
Report mean and standard deviation of accuracy across folds.
from sklearn.model_selection import cross_val_score
cv_scores_dt = cross_val_score(dt_shallow, X, y, cv=5)
cv_scores_rf = cross_val_score(rf, X, y, cv=5)

8. Visualization
Decision Tree: Use sklearn.tree.plot_tree() with max_depth=3 for clarity.
Feature Importance: Matplotlib bar chart.
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plot_tree(dt_shallow, feature_names=X.columns,
          class_names=['No Disease','Disease'], filled=True)
plt.show()

