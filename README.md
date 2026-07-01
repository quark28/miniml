# miniml

**Lightweight ML library from scratch. NumPy only.**

---

## 📦 Planned Features

### Models
- ~~Linear Regression~~
- Linear Classification
- Logistic Regression
- SVM
- PCA
- KNN Regression
- KNN Classification
- K-means
- DBSCAN
- Decision Tree Regression
- Decision Tree Classification
- Random Forest Regression
- Random Forest Classification
- Gradient Boosting Regression
- Gradient Boosting Classification
- Neural Networks
- The Levenberg-Marquardt Diagonal Method

### Linear Optimizers (Gradient Descent)
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