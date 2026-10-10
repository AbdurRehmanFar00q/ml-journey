# Linear Regression from Scratch: OLS (Normal Equation)

A NumPy-only implementation of multiple linear regression, solved in closed form with the normal equation and checked against scikit-learn.

## What this does

Linear regression predicts a number as a weighted sum of features:

```
y_hat = X @ w + b
```

Training finds the `w` (weights) and `b` (intercept) that minimize the mean squared error (MSE) on the training data. Ordinary Least Squares (OLS) solves this directly in one step, with no iteration.

## The math

Add a column of ones to `X` so the intercept is learned together with the weights. Call the new matrix `X_b` and the learned vector `beta = [b, w1, ..., wn]`. Setting the derivative of the MSE to zero gives the **normal equation**:

```
beta = (X_b^T X_b)^(-1) X_b^T y
```

The implementation uses `np.linalg.pinv` (pseudo-inverse) instead of a plain inverse. It gives the same answer when the matrix is invertible, and it still returns a sensible solution when features are perfectly correlated, where a plain inverse fails.

## Implementation

The class `LinearRegressionScratch` has three parts:

| Part | What it does |
|------|--------------|
| `fit(X, y)` | Adds the bias column, solves the normal equation, stores `intercept_` and `coef_` |
| `predict(X)` | Returns `X @ coef_ + intercept_` |
| `coef_`, `intercept_` | Learned parameters (trailing underscore follows the scikit-learn convention) |

## Usage

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11
)

model = LinearRegressionScratch()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Experiment

- **Dataset:** scikit-learn diabetes dataset (442 samples, 10 features)
- **Split:** 80% train / 20% test, `random_state=11`

| Metric (test set) | Value |
|-------------------|-------|
| R² score | 0.577 |
| Mean absolute error | 45.34 |

**Validation against scikit-learn:** fitting `sklearn.linear_model.LinearRegression` on the same training split gives the same coefficients and intercept as this implementation (`np.allclose` returned `True` for both), which confirms the implementation is correct.

## Limitations

- **Cost grows with the number of features.** Solving the normal equation takes roughly O(n_features^3) time, so it is impractical with a very large number of features. Gradient descent (next folder) scales better.
- **Sensitive to multicollinearity.** Highly correlated features make the solution numerically unstable, even with a pseudo-inverse.
- **Linear only.** The model cannot capture curved relationships without extra features (see polynomial regression, later in this series).

## Files

```
01-ols-from-scratch/
├── linear_regression.ipynb   # class, experiment, sklearn validation
└── README.md
```

## Run it

From the repository root, with the virtual environment active:

```bash
pip install -r requirements.txt
jupyter notebook 01-linear-regression/01-ols-from-scratch/linear_regression.ipynb
```

## What's next

Solve the same problem with gradient descent in `../02-gradient-descent-from-scratch/` and compare the two methods.