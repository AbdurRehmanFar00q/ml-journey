# Regularization: Ridge, Lasso, and ElasticNet

Regularization controls overfitting in linear models by adding a penalty on the size of the weights to the cost function. This folder contains Ridge regression built from scratch (closed form and gradient descent) and Ridge and Lasso with scikit-learn.

| Notebook | What it does |
|----------|--------------|
| `01-ridge-regression-scratch.ipynb` | Ridge from scratch with the closed-form solution |
| `02-ridge-regression-GD.ipynb` | Ridge from scratch with gradient descent |
| `03-ridge-regression-sklearn.ipynb` | `Ridge` and `Lasso` from scikit-learn |

## Why regularization

An overfit model, for example a high-degree polynomial, fits the training data very well and fails on new data. It usually does so with **very large weights** that cancel each other out to bend the curve through the noise. Regularization adds a price for large weights, so the model keeps a large weight only if it clearly improves the fit. The cost is a little more bias on the training data, and the gain is lower variance on new data.

## Ridge, Lasso, and ElasticNet

Let `ŷ` be the prediction, `w` the weights, and `α` (alpha) the penalty strength. With `α = 0` every model below is plain OLS.

| | Ridge (L2) | Lasso (L1) | ElasticNet |
|---|---|---|---|
| **Penalty** | sum of squared weights | sum of absolute weights | a mix of both |
| **Cost (scikit-learn form)** | `Σ(y − ŷ)² + α·Σw²` | `(1/2n)·Σ(y − ŷ)² + α·Σ|w|` | `(1/2n)·Σ(y − ŷ)² + α·ρ·Σ|w| + (α(1−ρ)/2)·Σw²` |
| **Effect on weights** | Shrinks all weights smoothly, rarely to exactly zero | Sets some weights to **exactly zero** | Shrinks, and zeroes some |
| **Feature selection** | No | Yes | Yes |
| **Closed-form solution** | Yes: `w = (XᵀX + αI)⁻¹ Xᵀy` | No (solved by coordinate descent) | No (coordinate descent) |
| **Hyperparameters** | `alpha` | `alpha` | `alpha` and `l1_ratio` (ρ) |
| **scikit-learn** | `Ridge`, `RidgeCV` | `Lasso`, `LassoCV` | `ElasticNet`, `ElasticNetCV` |

### Ridge
Squaring the weights means the penalty pulls hardest on large weights and barely at all on small ones, so weights shrink toward zero but seldom reach it. Adding `αI` to `XᵀX` also makes the matrix always invertible, which stabilizes the solution when features are strongly correlated.

**Use Ridge when:** you have many features that each contribute a little, features are correlated (multicollinearity), you want to keep every feature, or you want a stable, simple baseline.

### Lasso
The absolute-value penalty pulls on every non-zero weight with the same constant force, so weights that do not earn their place are driven to exactly zero (the mechanism is called soft thresholding). The result is a **sparse** model: Lasso performs feature selection and the model is easier to interpret.

**Use Lasso when:** many features are probably irrelevant, you want a smaller and more interpretable model, or you want built-in feature selection. With strongly correlated features it tends to keep one of them and drop the others somewhat arbitrarily, so the choice of which feature survives should not be over-interpreted.

### ElasticNet
ElasticNet combines the two penalties. `l1_ratio = 1` is Lasso and `l1_ratio = 0` is Ridge. It keeps Lasso's ability to zero out features while the Ridge part keeps groups of correlated features together instead of dropping all but one.

**Use ElasticNet when:** features are correlated and you also want sparsity, or you have more features than samples. Tune both `alpha` and `l1_ratio` (`ElasticNetCV` does this).

### Rules that apply to all three
1. **Standardize the features first** (or make sure they share a scale). The penalty compares weights directly, so features on a large scale would be penalized unfairly.
2. **Do not penalize the intercept.** Only the weights are shrunk.
3. **Choose `alpha` by cross-validation** on the training data, never by looking at the test score. Too small has no effect, and too large underfits.
4. **`alpha` is not comparable across models or datasets.** scikit-learn defines the Ridge and Lasso costs with different scaling, and the useful `alpha` also depends on the scale of the features.
5. **Ridge in gradient descent is "weight decay".** The penalty adds one term to the weight gradient, which shrinks the weights slightly at every step. The same idea is used in neural networks.

## Dataset and setup

- scikit-learn diabetes dataset (442 samples, 10 features), `load_diabetes(scaled=True)`
- 80% train / 20% test, `random_state=11` (the same split as the earlier folders)
- Metric: R² on the test set

## 1. Ridge from scratch: closed form

With a column of ones added to `X` for the intercept, the solution is `w = (XᵀX + αI)⁻¹ Xᵀy`. The first diagonal entry of `I` is set to 0, so the intercept is not penalized. The inverse is computed with `np.linalg.pinv`.

```python
I = np.eye(x.shape[1])
I[0,0] = 0  # Don't regularize the intercept
betas = np.linalg.pinv(np.dot(x.T,x) + self.alpha * I).dot(x.T).dot(y)
```

Result with `alpha=0.05`: **R² = 0.5700**.

## 2. Ridge from scratch: gradient descent

The cost is `(1/n)·Σ(y − ŷ)² + (α/n)·Σw²`. Compared with ordinary gradient descent, only the weight gradient gets a new term, and the intercept gradient is unchanged:

```
dJ/dw = −(2/n)·Σ(y − ŷ)·x + (2α/n)·w
dJ/db = −(2/n)·Σ(y − ŷ)
```

```python
coeff_der = -2 * np.dot((y-y_hat),x)/x.shape[0] + (2 * self.alpha/x.shape[0]) * self.coeff_
```

The penalty is divided by `n` so that `alpha` has the same meaning as in scikit-learn's `Ridge`.

Result with `alpha=0.01`, `lr=0.1`, `epoch=10000`: **R² = 0.5712**.

## 3. Ridge and Lasso with scikit-learn

```python
Ridge(alpha=0.1)   # R² = 0.5674
Lasso(alpha=0.1)   # R² = 0.5705
```

## Results

| Model | Settings | Test R² |
|-------|----------|--------:|
| OLS (earlier folder) | no penalty | 0.5772 |
| Ridge, closed form (from scratch) | alpha = 0.05 | 0.5700 |
| Ridge, gradient descent (from scratch) | alpha = 0.01, lr = 0.1, 10,000 epochs | 0.5712 |
| Ridge, scikit-learn | alpha = 0.1 | 0.5674 |
| Lasso, scikit-learn | alpha = 0.1 | 0.5705 |

### Effect on the coefficients

Printing the learned coefficients of the scikit-learn models (rounded) shows what each penalty does:

| Feature | OLS | Ridge (0.1) | Lasso (0.1) |
|---------|----:|------------:|------------:|
| `bmi` | 509 | 465 | 501 |
| `s1` | -450 | -45 | 0 |
| `s2` | 218 | -101 | -74 |
| `s4` | 56 | 76 | 0 |

Lasso sets two of the ten coefficients (`s1` and `s4`) to exactly zero, which Ridge does not do.

## Observations

- **On this dataset, regularization does not beat plain OLS.** Every regularized model scores slightly below OLS (0.5772). With only 10 features and 353 training samples there is little overfitting to fix, so the penalty has little to gain. Regularization matters most with many features or high-degree polynomial features.
- **Lasso zeroes features and Ridge does not.** This is the practical difference between the two. Lasso produces a sparser model.
- **Strongly correlated features change the most.** `s1` and `s2` (blood serum measurements that are highly correlated with each other) change a lot under regularization, while `bmi` barely moves. When features are correlated, OLS can assign them large opposite weights, and the penalty removes that instability.
- **The scratch implementations work.** The closed-form Ridge and the gradient descent Ridge give scores close to scikit-learn's and to each other.

## Limitations

- The `alpha` values were chosen by hand and not by cross-validation, and they differ between notebooks, so the Ridge scores are not strictly comparable with each other.
- Gradient descent was run on the default-scaled features, not on standardized ones, so it converges slowly. After 10,000 epochs the weights are still noticeably far from the exact Ridge solution, even though the R² looks close.
- ElasticNet is described above but not implemented in these notebooks.
- Results come from one train/test split.

## Files

```
03-regularization/
├── 01-ridge-regression-scratch.ipynb
├── 02-ridge-regression-GD.ipynb
├── 03-ridge-regression-sklearn.ipynb
└── README.md
```

## Run it

From the repository root, with the virtual environment active:

```bash
pip install -r requirements.txt
jupyter notebook 03-regularization/
```

