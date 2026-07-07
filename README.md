# miniml

**Lightweight ML library from scratch. NumPy only.**

---

## 📦 Planned Features

### Models and features
- ~~Linear Regression~~
- ~~Linear Classification~~
- ~~Scaler~~
- ~~Logistic Regression~~
- ~~SVM: linear~~
- ~~NBGaussianClassifier~~
- PCA
- KNN Regression
- KNN Classification
- K-means
- ~~DBSCAN~~
- Decision Tree Regression
- Decision Tree Classification
- Random Forest Regression
- Random Forest Classification
- Gradient Boosting Regression
- Gradient Boosting Classification
- Boostrap
- A/B Testing
- Naive bayes: Bernoulli, Multinomial
- Neural Networks module

### Future
- Gaussian Bayesian classifier
- The Levenberg-Marquardt Diagonal Method
- Fisher's linear discriminant
- Agglomerative hierarchical clustering
- SVM kernels: poly, rbf, tanh
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