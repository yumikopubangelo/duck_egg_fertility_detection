import numpy as np

from src.clustering.awc import AdaptiveWeightedClustering


def test_awc_feature_indices_are_applied_at_prediction_time():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 6)).astype(np.float32)
    X[:15, 1] -= 3.0
    X[15:, 1] += 3.0

    model = AdaptiveWeightedClustering(
        n_clusters=2,
        max_iter=20,
        feature_indices=[1, 4],
        random_state=42,
    )
    model.fit(X)

    assert model.centroids_.shape[1] == 2
    assert model.n_input_features_ == 6
    assert model.n_selected_features_ == 2

    full_preds = model.predict(X)
    selected_preds = model.predict(X[:, [1, 4]])
    assert np.array_equal(full_preds, selected_preds)

    proba = model.predict_proba(X[:3])
    assert proba.shape == (3, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
