# ENTRAP (Python) — Installation & Usage

ENTRAP (ENtropy-based Topological Rescue of Ambiguous Points) is a pure-Python library that refines HDBSCAN clustering results.

This is the **pure-Python** implementation of ENTRAP. Numerical hot paths (cluster mean, covariance, Mahalanobis distance, logistic mapping) are accelerated with **Numba JIT compilation** instead of Cython — no C compiler or `git` build step is required to install.

> Looking for the Cython build instead? See [`ENTRAP_Code/cython/README.md`](../cython/README.md).

---

## 1. Prerequisite

None beyond Python itself and `pip`. Unlike the Cython build, this package does **not** require Git or a C compiler — Numba compiles its JIT-decorated functions at **first call, at runtime**, not at install time.

> **Note:** the first call to a Numba-JIT function in a fresh process may take a few seconds longer than subsequent calls, since Numba compiles it to machine code on that first invocation and caches it for the remainder of the process.

---

## 2. Install ENTRAP

```bash
pip install git+https://github.com/AkmalHusain2003/ENTRAP_Python.git
```

To pin a specific commit (recommended for reproducibility):

```bash
pip install git+https://github.com/AkmalHusain2003/ENTRAP_Python.git@<commit-hash>
```

Requires **Python ≥ 3.8**. Dependencies (numpy, scikit-learn, scipy, numba, hdbscan, ripser, kneed, joblib, matplotlib) install automatically.

**Verify:**

```python
import entrap
print(entrap.__version__)   # -> "1.0.0"
```

---

## 3. Usage

Usage is **identical** to the Cython build — both expose the same `ENTRAP` class with the same public API, since they implement the same estimator.

### Basic

```python
import numpy as np
from entrap import ENTRAP

X = np.random.randn(500, 2)   # n_samples x n_features

model = ENTRAP(min_cluster_size=30, metric='euclidean')
model.fit(X)

labels = model.labels_        # -1 = noise, same convention as HDBSCAN
print(model.get_summary())
```

Or directly:

```python
labels = model.fit_predict(X)
```

### `ENTRAP` parameters

| Parameter | Default | Description |
|---|---|---|
| `min_cluster_size` | `30` | Passed to HDBSCAN. |
| `min_samples` | `None` | Passed to HDBSCAN; defaults to `min_cluster_size` if `None`. |
| `ridge_epsilon` | `1e-6` | Internal covariance regularization. |
| `metric` | `'euclidean'` | See supported metrics below. |
| `metric_params` | `None` | Extra dict params for the chosen metric (e.g. `{'p': 3}`). |
| `use_memmap` | `True` | Stored as an attribute; has no effect on `fit()` behavior in this version. |
| `enable_tracking` | `False` | If `True`, enables diagnostic tracking (see below). |
| `n_jobs` | `-1` | Parallel workers. `-1` = all CPU cores. |

### Supported metrics

```
euclidean, manhattan, cityblock, minkowski, chebyshev,
cosine, correlation, hamming, jaccard, canberra,
braycurtis, mahalanobis, seuclidean, sqeuclidean
```

A custom Python callable is also accepted.

### Accessing full results

```python
result = model.result_

result.labels             # same as model.labels_
result.probabilities      # HDBSCAN cluster membership probabilities
result.noise_rescued      # number of rescued noise points
result.execution_time     # seconds
result.n_clusters         # final cluster count
result.cluster_stats      # per-cluster stats
result.hyperparameters    # hyperparameters used
result.tracker            # tracker object, only if enable_tracking=True
```

### Diagnostics (optional)

```python
model = ENTRAP(min_cluster_size=30, enable_tracking=True)
model.fit(X)

model.list_tracked_clusters()
model.plot_entropy_curve(cluster_id=0)
model.plot_rejected_analysis(cluster_id=0)
model.plot_comparison(cluster_ids=[0, 1, 2])

accepted = model.get_accepted_candidates(cluster_id=0)
rejected = model.get_rejected_candidates(cluster_id=0)
summary = model.export_entropy_summary()
```

All `plot_*` methods accept an optional `save_path` to save the figure.

> Calling these without `enable_tracking=True` at `fit()` time raises a `ValueError`.

---

## 4. Cython vs. Python build

| | Python (this package) | Cython |
|---|---|---|
| Install-time requirement | None (pure Python) | Git + C compiler |
| Numerical acceleration | Numba JIT (`@njit`, compiled on first call) | Cython (compiled to C at install time) |
| First-run overhead | A few seconds of JIT warm-up on first call | None — already compiled at install |
| Extra dependency | `numba>=0.56.0` | `cython>=3.0.0` |

Both builds implement the same `ENTRAP` estimator and produce equivalent results; choose based on whether a build toolchain is available in your environment.

---

## 5. Install Troubleshooting

**`ImportError: numba requires ...` / Numba fails to install**
→ Numba requires a supported combination of Python and NumPy versions. Check the [Numba compatibility table](https://numba.readthedocs.io/en/stable/user/installing.html) and, if needed, upgrade/downgrade NumPy to a supported range.

**First `fit()` call is noticeably slower than later calls**
→ Expected — Numba is JIT-compiling the decorated functions on first use in that process. Subsequent calls in the same process reuse the compiled code.

**`git: command not found`**
→ Only needed for `pip install git+...`. See your platform's Git installation instructions (e.g. `sudo apt install git`, `brew install git`, or [git-scm.com](https://git-scm.com/download/win)).
