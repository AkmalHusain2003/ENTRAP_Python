import numpy as np
from sklearn.linear_model import LinearRegression


def estimate_intrinsic_dimension_twenn(
    X: np.ndarray,
    X_is_dist: bool = False
) -> float:
    N = X.shape[0]
    if N < 3:
        return 1.0

    if X_is_dist:
        dist = X
    else:
        from scipy.spatial.distance import squareform, pdist
        dist = squareform(pdist(X, metric='euclidean'))

    mu = np.zeros(N)
    for i in range(N):
        sort_idx = np.argsort(dist[i, :])
        r1 = dist[i, sort_idx[1]]
        r2 = dist[i, sort_idx[2]]
        mu[i] = r2 / r1 if r1 > 1e-12 else 1.0

    sort_idx = np.argsort(mu)
    Femp = np.arange(N) / N

    lr = LinearRegression(fit_intercept=False)
    log_mu = np.log(mu[sort_idx] + 1e-12).reshape(-1, 1)
    neg_log_1_minus_F = -np.log(1 - Femp + 1e-12).reshape(-1, 1)
    lr.fit(log_mu, neg_log_1_minus_F)

    d_hat = lr.coef_[0][0]
    max_dim = X.shape[1] if not X_is_dist else N
    return np.clip(d_hat, 1.0, max_dim)
