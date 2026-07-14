# miniml

**Lightweight ML library from scratch. NumPy only.**

---

## 📁 Structure
```text
miniml/
├── linear/
│   ├── init.py
│   ├── linear_regression.py
│   ├── linear_classification.py
│   ├── logistic_regression.py
│   └── svm.py
│
├── cluster/
│   ├── init.py
│   ├── dbscan.py
│   └── kmeans.py
│
├── probability/
│   ├── init.py
│   ├── nbgaussian.py
│   ├── nbbernoulli.py
│   └── nbmultinomial.py
│
├── tree_based/
│   ├── init.py
│   ├── decisiontreeclassifier.py
│   ├── decisiontreeregressor.py
│   ├── randomforestclassifier.py
│   ├── randomforestregressor.py
│   ├── gradientboostingregression.py
│   └── gradientboostingbinaryclassification.py
│
├── knn/
│   ├── init.py
│   ├── knn_classifier.py
│   └── knn_regressor.py
│
├── tools/
│   ├── init.py
│   ├── scaler.py
│   ├── pca.py
│   └── bagging.py
│
├── metrics/
│   ├── init.py
│   ├── classification_metrics.py
│   └── regression_metrics.py
│
├── tests/
│   └── ...  # Jupyter notebooks with model testing
│
├── init.py
├── requirements.txt
├── LICENSE
└── README.md
```
---

## 📦 Implemented

### Models
- Linear Regression
- Linear Classification
- Logistic Regression
- SVM: linear
- NBGaussianClassifier
- NBBernoulliClassifier
- NBMultinomialClassifier
- KNN Regression
- KNN Classification
- K-means
- DBSCAN
- Decision Tree Regression
- Decision Tree Classification
- Random Forest Regression
- Random Forest Classification
- Gradient Boosting Regression
- GradientBoostingBinaryClassification

### Tools
- Scaler
- PCA
- Bootstrap

### Metrics
- Classification metrics (accuracy, precision, recall, F-score, ROC-AUC, PR-AUC)
- Regression metrics

---

## 🔮 Future

- Neural Networks module
- A/B Testing
- Gaussian Bayesian classifier
- The Levenberg-Marquardt Diagonal Method
- Fisher's linear discriminant
- Agglomerative hierarchical clustering
- SVM kernels: poly, rbf, tanh
- GD's as multi-class tools
- Optimization
- GradientBoostingMulticlassClassification

---

## ⚙️ Linear Optimizers (Gradient Descents)

Shared across all linear models (`LinearRegression`, `LinearClassification`, `LogisticRegression`, `SVM`) via a unified `fit(learning_type=...)` interface.

- Classic GD
- SGD (Stochastic Gradient Descent)
- SAG (Stochastic Average Gradient)
- Momentum
- NAG – Nesterov's Accelerated Gradient
- AdaGrad
- AdaDelta (adaptive learning rate)
- RMSprop
- Adam
- Nadam

## 📐 Analytical Solvers (LinearRegression, MSE only)

- Classic (Normal Equation)
- SVD

---

## License

See [LICENSE](LICENSE).