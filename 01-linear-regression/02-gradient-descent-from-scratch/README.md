# Linear Regression from Scratch: Gradient Descent

A NumPy-only implementation of multiple linear regression trained with batch gradient descent, tested on the scikit-learn diabetes dataset.

## What this does

Instead of solving the normal equation in one step (see `../01-ols-from-scratch/`), gradient descent starts from an initial guess and repeatedly moves the weights and intercept in the direction that lowers the mean squared error (MSE).

## The math

Model and cost:

```
y_hat_i = w . x_i + b
J(w, b) = (1/n) * sum_i (y_i - y_hat_i)^2
```

Derivatives, using the chain rule:

```
dJ/dw_j = -(2/n) * sum_i (y_i - y_hat_i) * x_ij
dJ/db   = -(2/n) * sum_i (y_i - y_hat_i)
```

Update rule, repeated for `epoch` iterations:

```
w <- w - lr * dJ/dw
b <- b - lr * dJ/db
```

`lr` is the learning rate, which controls the step size.

## Implementation

`LinearRegressionGD(lr=0.01, epoch=1000)`

| Part | What it does |
|------|--------------|
| `fit(x, y)` | Initializes `coeff_` to ones and `intercept_` to 0, then runs `epoch` gradient steps |
| `predict(x)` | Returns `intercept_ + x @ coeff_` |
| `coeff_`, `intercept_` | Learned weights and bias |
| `loss_history_` | Training MSE recorded at every iteration |

In each iteration the predictions `y_hat` are computed once and used for both gradients, so the weights and the intercept are updated from the same point.

## Usage

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

x, y = load_diabetes(return_X_y=True, scaled=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=11
)

model = LinearRegressionGD(lr=0.1, epoch=10000)
model.fit(x_train, y_train)
pred = model.predict(x_test)

print("R2 score: ", r2_score(y_test, pred))
```

## Experiment

- **Dataset:** scikit-learn diabetes (442 samples, 10 features)
- **Features:** `load_diabetes(scaled=True)`, the default scaling. Each column has a standard deviation of about 0.0476 (printed by `x.std(axis=0)`), so the features are small in scale but not unit variance.
- **Split:** 80% train / 20% test, `random_state=11`
- **Hyperparameters:** `lr=0.1`, `epoch=10000`

### Result

| Metric | Value |
|--------|-------|
| Test R² (gradient descent) | 0.5714 |
| Test R² (OLS, same split, from the OLS notebook) | 0.5772 |

The training loss starts at about 28,486 (with weights initialized to ones) and falls steeply in the first few iterations.

### Loss curve

![Loss curve](loss_curve.png)

The curve, drawn on a log scale, shows a steep initial drop followed by a long, slow decline. It flattens out after roughly 5,000 to 6,000 iterations, and it has no spikes, which means the learning rate of 0.1 is stable for this data.

## Observations

- The implementation works: the loss decreases smoothly and the model reaches a test R² close to the OLS result.
- Gradient descent scores slightly below OLS (0.5714 vs 0.5772). OLS gives the exact minimum of the training error, while gradient descent only approaches it, so a small gap is expected when the optimization has not fully converged.
- Most of the improvement happens in the first few hundred iterations. The remaining iterations change the loss very little.

## Limitations and next steps

- The number of epochs is fixed. A stopping rule based on a small change in loss would avoid wasted iterations.
- Convergence was judged by test R² and the loss curve only. A stronger check is to compare the learned `coeff_` and `intercept_` with `sklearn.linear_model.LinearRegression`.
- Features were not standardized with `StandardScaler`. Standardizing usually makes gradient descent converge faster and allows a wider range of learning rates.
- Only batch gradient descent is implemented. Stochastic and mini-batch variants are described in the next section but are not implemented in this notebook.

## Extending this code to stochastic and mini-batch gradient descent

This section only describes the changes. It is not implemented or tested in this notebook.

All three variants are the same algorithm. The only difference is how many samples are used to compute each update, which is the **batch size**.

| Variant | `batch_size` | Updates per epoch | Behavior |
|---------|-------------:|------------------:|----------|
| Batch (this notebook) | all `n` samples | 1 | Smooth loss curve, but each step is costly on large data |
| Stochastic (SGD) | 1 | `n` | Cheap steps, very noisy path that keeps wobbling around the minimum |
| Mini-batch | for example 32 | `n / batch_size` | A compromise, and the standard choice for neural networks |

### What to change in `LinearRegressionGD`

1. **`__init__`:** add a `batch_size` parameter and store it as `self.batch_size`.
2. **`fit`, each epoch:** shuffle the data with `np.random.permutation`, so the batches differ from epoch to epoch.
3. **`fit`, new inner loop:** go through the shuffled data in slices of `batch_size`, giving `x_batch` and `y_batch`.
4. **Gradients:** compute `y_hat` and both derivatives on the batch only. Replace `x.shape[0]` with `x_batch.shape[0]`, because the average must be over the batch size.
5. **Update after every batch,** so the two update lines move inside the inner loop.
6. **`loss_history_`:** record the loss on the full training data once per epoch, after the inner loop.

Only `__init__` and `fit` change. `predict` stays the same.

```python
def __init__(self, lr=0.01, epoch=1000, batch_size=32):
    self.lr = lr
    self.epoch = epoch
    self.batch_size = batch_size
    self.coeff_= None
    self.intercept_ = None

def fit(self,x,y):
    self.coeff_ = np.ones(x.shape[1])
    self.intercept_ = 0
    self.loss_history_ = []

    for i in range(self.epoch):
        idx = np.random.permutation(x.shape[0])
        x_shuffled = x[idx]
        y_shuffled = y[idx]

        for j in range(0, x.shape[0], self.batch_size):
            x_batch = x_shuffled[j:j+self.batch_size]
            y_batch = y_shuffled[j:j+self.batch_size]

            y_hat = np.dot(x_batch, self.coeff_) + self.intercept_
            intercept_der = -2 * np.mean(y_batch-y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)

            coeff_der = -2 * np.dot((y_batch-y_hat),x_batch)/x_batch.shape[0]
            self.coeff_ = self.coeff_ - (self.lr * coeff_der)

        y_hat_full = np.dot(x, self.coeff_) + self.intercept_
        self.loss_history_.append(np.mean((y-y_hat_full)**2))
```

How to choose the variant when creating the model:

```python
model = LinearRegressionGD(lr=0.1, epoch=500, batch_size=x_train.shape[0])  # batch
model = LinearRegressionGD(lr=0.01, epoch=100, batch_size=32)               # mini-batch
model = LinearRegressionGD(lr=0.001, epoch=30, batch_size=1)                # stochastic
```

### Things to keep in mind

- **`epoch` changes meaning.** In batch gradient descent one epoch is one update. In SGD and mini-batch one epoch is one pass over the data, which contains many updates, so far fewer epochs are needed.
- **Use a smaller learning rate for SGD.** A learning rate that is stable for batch gradient descent can make SGD diverge, because single-sample gradients are noisy.
- **Expect noisy loss curves.** SGD and mini-batch curves do not settle exactly. They hover around the minimum.
- **Standardize the features first,** exactly as for batch gradient descent.

### How to check it works

- With `batch_size=x_train.shape[0]` the results should match the current batch version.
- Compare `coeff_` with `sklearn.linear_model.LinearRegression` for each batch size.
- Plot `loss_history_` for batch, mini-batch, and SGD on one chart.

## Files

```
02-gradient-descent-from-scratch/
├── linearRegressionGD.ipynb   # class, training, evaluation, loss plot
├── loss_curve.png             # saved from the notebook
└── README.md
```

## Run it

From the repository root, with the virtual environment active:

```bash
pip install -r requirements.txt
jupyter notebook 01-linear-regression/02-gradient-descent-from-scratch/linearRegressionGD.ipynb
```

