# miniml

**Lightweight ML library from scratch. NumPy only.**

---
## 📁 Structure
miniml/
├── linear/
│ ├── init.py
│ ├── linear_regression.py
│ ├── linear_classification.py
│ └── logistic_regression.py
│ └── svm.py
│
├── cluster/
│ ├── init.py
│ └── dbscan.py
│ └── kmeans.py
│
├── probability/
│ ├── init.py
│ └── naive_bayes.py
│
├── tools/
│ ├── init.py
│ └── scaler.py
│
├── metrics/
│ ├── init.py
│ ├── classification_metrics.py
│ └── regression_metrics.py
│
├── knn/
│ ├── init.py
│ ├── knn_regressor.py
│ └── knn_classifier.py
│
├── tests/
├── init.py
├── requirements.txt
├── README.md
└── .gitignore

## 📦 Planned Features

### Models and features
- ~~Linear Regression~~
- ~~Linear Classification~~
- ~~Scaler~~
- ~~Logistic Regression~~
- ~~SVM: linear~~
- ~~NBGaussianClassifier~~
- ~~NBBernoulliClassifier~~
- ~~NBMultinomialClassifier~~
- ~~PCA~~
- ~~KNN Regression~~
- ~~KNN Classification~~
- ~~K-means~~
- ~~DBSCAN~~
- Boostrap
- ~~Decision Tree Regression~~
- ~~Decision Tree Classification~~
- ~~Random Forest Regression~~
- ~~Random Forest Classification~~
- Gradient Boosting Regression
- Gradient Boosting Classification

### Future
- Neural Networks module
- A/B Testing
- Gaussian Bayesian classifier
- The Levenberg-Marquardt Diagonal Method
- Fisher's linear discriminant
- Agglomerative hierarchical clustering
- SVM kernels: poly, rbf, tanh
- GD's as multi-class tools
- Optimization

### Linear Optimizers (Gradient Descents)
- ~~Classic GD~~
- ~~SGD (Stochastic Gradient Descent)~~
- ~~SAG (Stochastic Average Gradient)~~
- ~~momentum~~
- ~~NAG – Nesterov’s accelerated gradient~~
- ~~AdaDelta (adaptive learning rate)~~
- ~~AdaGrad~~
- ~~Adam~~
- ~~Nadam~~
- ~~RMSprop~~

### Analytical Solvers
- ~~Classic (Normal Equation)~~
- ~~SVD~~

---

## 🚀 Getting Started

```python
import numpy as np
from miniml import LinearRegression

# Generate sample data
X = np.random.randn(100, 3)
y = X @ np.array([2, -1, 0.5]) + 3

# Train model
model = LinearRegression(loss='mse', regularizer='ridge', alpha=0.1)
model.fit(X, y, optimizer='gd', n_epochs=1000)

# Predict
predictions = model.predict(X)