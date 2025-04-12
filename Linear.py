import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 5)  # 5 independent variables
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100)  # Only first 2 features matter

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression
ridge = Ridge(alpha=1.0)  # λ = 1
ridge.fit(X_train, y_train)
ridge_preds = ridge.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_preds)
print("Ridge Coefficients:", ridge.coef_)
print("Ridge MSE:", ridge_mse)

# Lasso Regression
lasso = Lasso(alpha=0.1)  # λ = 0.1
lasso.fit(X_train, y_train)
lasso_preds = lasso.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_preds)
print("Lasso Coefficients:", lasso.coef_)
print("Lasso MSE:", lasso_mse)

# Plotting Coefficients
plt.bar(range(len(ridge.coef_)), ridge.coef_, color="blue", alpha=0.5, label="Ridge")
plt.bar(range(len(lasso.coef_)), lasso.coef_, color="red", alpha=0.5, label="Lasso")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.legend()
plt.show()
