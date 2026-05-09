import numpy as np
from miniml import LinearRegression

np.random.seed(42)
X = np.random.randn(100, 3)
true_coef = np.array([2, -1, 0.5])
y = X @ true_coef + 3 + np.random.randn(100) * 0.1

# Classic
model = LinearRegression(loss='mse', regularizator='ridge', alpha_reg=0.1)
model.fit(X, y, learning_type='as', ad_type='classic')

print(f"Coefficients: {model.coefficients}")
print(f"Bias: {model.bias:.6f}")
