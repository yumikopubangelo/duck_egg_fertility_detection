"""Fuzzy C-Means Baseline Classifier for Duck Egg Fertility Detection."""
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class FuzzyCMeans:
    def __init__(self, c=2, m=2.0, error=0.005, max_iter=1000,
                 random_state=42, scale_features=True):
        self.c = c
        self.m = m
        self.error = error
        self.max_iter = max_iter
        self.random_state = random_state
        self._scaler = StandardScaler() if scale_features else None
        self._centroids = None
        self._cluster_to_label = {}
        self.n_iter_ = 0
        self.silhouette_ = 0.0
        self.is_fitted_ = False

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)
        U, centroids, n_iter = self._fcm_fit(X)
        self._centroids = centroids
        self.n_iter_ = n_iter
        hard = np.argmax(U, axis=0)
        if len(np.unique(hard)) > 1:
            self.silhouette_ = float(silhouette_score(X, hard, sample_size=min(500, len(X))))
        if y is not None:
            y = np.asarray(y)
            self._cluster_to_label = {
                cid: int(np.bincount(y[hard == cid].astype(int)).argmax())
                for cid in np.unique(hard)
            }
        else:
            means = {cid: float(X[hard == cid].mean()) for cid in np.unique(hard)}
            sc = sorted(means, key=means.get)
            self._cluster_to_label = {sc[0]: 0, sc[1]: 1}
        self.is_fitted_ = True
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        U = self._compute_membership(X, self._centroids)
        proba = np.zeros((X.shape[0], 2), dtype=np.float32)
        for cid, lbl in self._cluster_to_label.items():
            proba[:, lbl] += U[cid]
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return proba / row_sums

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _fcm_fit(self, X):
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        U = rng.random((self.c, n))
        U /= U.sum(axis=0, keepdims=True)
        centroids = None
        for it in range(self.max_iter):
            U_prev = U.copy()
            Um = U ** self.m
            centroids = (Um @ X) / Um.sum(axis=1, keepdims=True)
            U = self._compute_membership(X, centroids)
            if np.linalg.norm(U - U_prev) < self.error:
                return U, centroids, it + 1
        return U, centroids, self.max_iter

    def _compute_membership(self, X, centroids):
        dists = np.array([np.linalg.norm(X - centroids[k], axis=1)
                          for k in range(self.c)])
        dists = np.maximum(dists, 1e-10)
        exp = 2.0 / (self.m - 1)
        ratio = dists[:, None, :] / dists[None, :, :]
        return 1.0 / (ratio ** exp).sum(axis=1)
