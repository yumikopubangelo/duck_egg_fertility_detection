"""K-Means Baseline Classifier for Duck Egg Fertility Detection."""
import numpy as np
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class KMeansBaseline:
    def __init__(self, n_clusters=2, n_init=10, max_iter=300,
                 random_state=42, scale_features=True):
        self.n_clusters = n_clusters
        self._kmeans = KMeans(n_clusters=n_clusters, init="k-means++",
                              n_init=n_init, max_iter=max_iter, random_state=random_state)
        self._scaler = StandardScaler() if scale_features else None
        self._cluster_to_label = {}
        self.inertia_ = 0.0
        self.silhouette_ = 0.0
        self.is_fitted_ = False

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)
        self._kmeans.fit(X)
        raw = self._kmeans.labels_
        self.inertia_ = float(self._kmeans.inertia_)
        if len(np.unique(raw)) > 1:
            self.silhouette_ = float(silhouette_score(X, raw, sample_size=min(500, len(X))))
        if y is not None:
            y = np.asarray(y)
            self._cluster_to_label = {
                cid: int(np.bincount(y[raw == cid].astype(int)).argmax())
                for cid in np.unique(raw)
            }
        else:
            means = {cid: float(X[raw == cid].mean()) for cid in np.unique(raw)}
            sc = sorted(means, key=means.get)
            self._cluster_to_label = {sc[0]: 0, sc[1]: 1}
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        raw = self._kmeans.predict(X)
        return np.array([self._cluster_to_label.get(c, 0) for c in raw], dtype=int)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self._scaler is not None:
            X = self._scaler.transform(X)
        dists = np.linalg.norm(X[:, None, :] - self._kmeans.cluster_centers_[None, :, :], axis=2)
        neg = -dists
        exp_ = np.exp(neg - neg.max(axis=1, keepdims=True))
        soft = exp_ / exp_.sum(axis=1, keepdims=True)
        proba = np.zeros((len(X), 2), dtype=np.float32)
        for cid, lbl in self._cluster_to_label.items():
            proba[:, lbl] += soft[:, cid]
        return proba

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
