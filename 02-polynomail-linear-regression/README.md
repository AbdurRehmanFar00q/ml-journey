# Polynomial Regression

Fitting a curve with linear regression by adding powers of the input as new features, compared against a plain straight-line model on the same data.

## The idea

A straight line cannot follow curved data. Polynomial regression solves this without changing the algorithm. It creates new features from powers of `x`, then fits an ordinary linear regression on them:

```
y_hat = b + w1*x + w2*x^2 + w3*x^3
```

The model is still **linear in the weights** (`w1`, `w2`, `w3`), even though it is curved in `x`. So the same least-squares fit used in the earlier lessons applies. Only the input matrix changes:

```
x  ->  [x, x^2, x^3]
```

For example, one training value from the notebook:

```
x          = [0.53613198]
x expanded = [0.53613198, 0.2874375, 0.15410443]
```

## Dataset

Synthetic data with a known curve, 200 samples:

```python
x = 6*np.random.rand(200,1) - 3                          # uniform between -3 and 3
y = 0.8 * x**2 + 0.9 * x + 2 + np.random.randn(200,1)    # quadratic + noise (std 1)
```

The true relationship is a quadratic, so the data has a clear curve.

## Method

1. Split the data: 80% train, 20% test, `random_state=11`.
2. Create polynomial features with `PolynomialFeatures(degree=3, include_bias=False)`, **fitting on the training data only** and applying `transform` to the test data (this avoids data leakage).
3. Fit `LinearRegression` on the expanded training features and predict on the expanded test features.
4. Fit a second `LinearRegression` on the original `x` as the baseline.
5. Plot both models. To draw a smooth curve, predict on 200 evenly spaced x values from -3 to 3, passing them through the same `poly_features.transform` first.

## Results

| Model | Test R² |
|-------|--------:|
| Polynomial regression (degree 3) | 0.8818 |
| Plain linear regression | 0.3289 |

![Polynomial regression fit](polynomial_fit.png)

![Linear regression fit](linear_fit.png)

## Observations

- **The straight line underfits.** It scores R² 0.3289 because it cannot follow the curvature of the data.
- **Polynomial features fix this.** The same linear regression, given `x`, `x^2`, and `x^3`, reaches R² 0.8818 and its curve follows the data.
- **R² is close to the best possible value here.** Because the data contains random noise with standard deviation 1, even a model that knew the true formula could not score above about 0.876 (the share of the variance that is signal and not noise). The result is essentially at that limit, and the small excess over it comes from the test set being only 40 samples.
- **"Linear" regression can fit curves.** The model is linear in its weights, not in `x`, which is why the least-squares machinery still works.

## Limitations

- The data is generated **without a fixed random seed**, so the numbers above are from one run. Re-running the notebook gives slightly different values.
- Degree 3 was chosen without comparing other degrees. The true curve is a quadratic, so degree 2 should perform about as well with fewer parameters. A sweep over degrees, comparing train and test scores, would show where overfitting begins.
- Results come from a single train/test split.
- Only scikit-learn's `PolynomialFeatures` was used, with no from-scratch version. No feature scaling was needed here because `x` stays between -3 and 3, but higher degrees need scaling.

## Files

```
02-polynomial-regression/
├── polynomial-regression.ipynb
├── polynomial_fit.png
├── linear_fit.png
└── README.md
```

## Run it

From the repository root, with the virtual environment active:

```bash
pip install -r requirements.txt
jupyter notebook 02-polynomial-regression/polynomial-regression.ipynb
```

