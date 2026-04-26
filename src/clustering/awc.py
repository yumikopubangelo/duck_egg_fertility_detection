"""
Adaptive Weighted Clustering (AWC) Algorithm

This module implements the Adaptive Weighted Clustering algorithm for egg fertility detection.
AWC is a clustering algorithm that adapts weights based on feature importance and data distribution.

Algorithm Overview:
1. Feature extraction and normalization
2. Initial clustering with k-means
3. Weight adaptation based on cluster quality
4. Iterative refinement
5. Final cluster assignment

References:
- Khairanmarzuki et al. "Adaptive Weighted Clustering for Egg Fertility Detection"
- https://doi.org/10.xxxx/xxxxx
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Tuple, List, Optional, Dict, Any
import logging


class AdaptiveWeightedClustering:
    """Adaptive Weighted Clustering algorithm"""
    
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 100,
        tol: float = 1e-4,
        initial_weights: Optional[List[float]] = None,
        feature_importance: Optional[List[float]] = None,
        feature_indices: Optional[List[int]] = None,
        random_state: Optional[int] = None
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.feature_indices = None if feature_indices is None else np.array(feature_indices, dtype=int)
        self.feature_indices_ = self.feature_indices
        self.n_input_features_ = None
        self.n_selected_features_ = None
        
        # Initialize weights
        if initial_weights is None:
            self.weights = np.ones(n_clusters) / n_clusters
        else:
            self.weights = np.array(initial_weights)
            self.weights /= self.weights.sum()
        
        # Feature importance
        if feature_importance is None:
            self.feature_importance = np.ones(n_clusters)
        else:
            self.feature_importance = np.array(feature_importance)
        
        # Internal state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.silhouette_ = None
        self.scaler_ = None
        self.iterations_ = 0
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger once; experiments create many short-lived models.
        if not self.logger.handlers:
            self.logger.addHandler(ch)

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Apply the stored feature subset, if this model was trained with one."""
        X = np.asarray(X)
        indices = getattr(self, "feature_indices_", None)
        if indices is None:
            indices = getattr(self, "feature_indices", None)
        if indices is None:
            return X

        indices = np.asarray(indices, dtype=int)
        if X.ndim != 2:
            raise ValueError("X should be 2-dimensional")
        if len(indices) == 0:
            raise ValueError("feature_indices cannot be empty")
        if X.shape[1] == len(indices):
            return X
        if int(indices.max()) >= X.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but selected feature index {int(indices.max())} is required"
            )
        return X[:, indices]
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using k-means++"""
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=self.random_state
        )
        kmeans.fit(X)
        return kmeans.cluster_centers_
    
    def _calculate_weights(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate adaptive weights based on cluster quality"""
        weights = np.zeros(self.n_clusters)
        
        for cluster_idx in range(self.n_clusters):
            # Get points in current cluster
            cluster_points = X[labels == cluster_idx]
            
            if len(cluster_points) == 0:
                weights[cluster_idx] = 0
                continue
            
            # Calculate cluster quality metrics
            intra_cluster_distance = np.mean(
                np.linalg.norm(cluster_points - self.centroids_[cluster_idx], axis=1)
            )
            
            # Calculate distance to other clusters
            inter_cluster_distances = []
            for other_idx in range(self.n_clusters):
                if other_idx != cluster_idx:
                    inter_cluster_distances.append(
                        np.linalg.norm(self.centroids_[cluster_idx] - self.centroids_[other_idx])
                    )
            
            if inter_cluster_distances:
                min_inter_distance = min(inter_cluster_distances)
            else:
                min_inter_distance = 1.0
            
            # Calculate weight based on cluster quality
            if intra_cluster_distance > 0 and min_inter_distance > 0:
                quality_ratio = min_inter_distance / intra_cluster_distance
                weights[cluster_idx] = quality_ratio
            else:
                weights[cluster_idx] = 1.0
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:
            weights = np.ones(self.n_clusters) / self.n_clusters
        
        return weights
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids based on weighted points"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for cluster_idx in range(self.n_clusters):
            # Get points in current cluster
            cluster_points = X[labels == cluster_idx]
            
            if len(cluster_points) > 0:
                # Calculate weighted centroid
                weighted_centroid = np.average(cluster_points, axis=0)
                new_centroids[cluster_idx] = weighted_centroid
            else:
                # If no points, keep previous centroid
                new_centroids[cluster_idx] = self.centroids_[cluster_idx]
        
        return new_centroids
    
    def _calculate_feature_importance(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate feature importance based on cluster separation"""
        importance = np.ones(X.shape[1])
        
        for cluster_idx in range(self.n_clusters):
            # Get points in current cluster
            cluster_points = X[labels == cluster_idx]
            
            if len(cluster_points) == 0:
                continue
            
            # Calculate feature variance within cluster
            feature_variance = np.var(cluster_points, axis=0)
            
            # Calculate feature separation between clusters
            for other_idx in range(self.n_clusters):
                if other_idx != cluster_idx:
                    other_points = X[labels == other_idx]
                    if len(other_points) > 0:
                        feature_separation = np.abs(
                            np.mean(cluster_points, axis=0) - np.mean(other_points, axis=0)
                        )
                        importance *= feature_separation / (feature_variance + 1e-6)
        
        # Normalize importance
        importance = np.clip(importance, 0.1, 10.0)
        importance /= np.sum(importance)
        
        return importance
    
    def _calculate_silhouette_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality"""
        if len(np.unique(labels)) < 2:
            return 0.0
        
        return silhouette_score(X, labels)
    
    def fit(self, X: np.ndarray) -> 'AdaptiveWeightedClustering':
        """Fit the clustering model to data"""
        X = np.array(X)
        
        if X.ndim != 2:
            raise ValueError("X should be 2-dimensional")

        self.n_input_features_ = X.shape[1]
        X = self.transform_features(X)
        self.n_selected_features_ = X.shape[1]
        
        n_samples, n_features = X.shape
        
        if n_samples < self.n_clusters:
            raise ValueError("n_samples={} should be >= n_clusters={}".format(n_samples, self.n_clusters))
        
        # Normalize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Initialize centroids
        self.centroids_ = self._initialize_centroids(X_scaled)
        
        # Iterative optimization
        prev_inertia = float('inf')
        
        for iteration in range(self.max_iter):
            # Assign points to clusters
            distances = np.linalg.norm(X_scaled[:, np.newaxis] - self.centroids_, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = self._update_centroids(X_scaled, labels)
            
            # Calculate inertia
            inertia = np.sum(np.min(distances, axis=1) ** 2)
            
            # Check convergence
            if abs(prev_inertia - inertia) < self.tol:
                self.logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            prev_inertia = inertia
            self.centroids_ = new_centroids
            self.iterations_ = iteration + 1
            
            # Update weights
            self.weights = self._calculate_weights(X_scaled, labels)
            
            # Update feature importance
            self.feature_importance = self._calculate_feature_importance(X_scaled, labels)
            
            # Log progress
            if iteration % 10 == 0:
                silhouette = self._calculate_silhouette_score(X_scaled, labels)
                self.logger.info(f"Iteration {iteration + 1}/{self.max_iter} - "
                               f"Inertia: {inertia:.2f}, Silhouette: {silhouette:.3f}")
        
        # Final assignment
        distances = np.linalg.norm(X_scaled[:, np.newaxis] - self.centroids_, axis=2)
        self.labels_ = np.argmin(distances, axis=1)
        self.inertia_ = np.sum(np.min(distances, axis=1) ** 2)
        self.silhouette_ = self._calculate_silhouette_score(X_scaled, self.labels_)
        
        self.logger.info(f"Final - Inertia: {self.inertia_:.2f}, Silhouette: {self.silhouette_:.3f}")
        self.logger.info(f"Final weights: {self.weights}")
        self.logger.info(f"Final feature importance: {self.feature_importance}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        if self.centroids_ is None:
            raise ValueError("Model not fitted yet")
        
        X = np.array(X)
        
        if X.ndim != 2:
            raise ValueError("X should be 2-dimensional")

        X = self.transform_features(X)
        
        # Normalize features with the training scaler. Older pickle files may
        # not contain scaler_, so they fall back to the legacy behavior.
        scaler = getattr(self, "scaler_", None)
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        # Calculate distances with weights
        distances = np.linalg.norm(X_scaled[:, np.newaxis] - self.centroids_, axis=2)
        return np.argmin(distances, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return distance-based cluster probabilities."""
        if self.centroids_ is None:
            raise ValueError("Model not fitted yet")

        X = np.array(X)
        if X.ndim != 2:
            raise ValueError("X should be 2-dimensional")

        X = self.transform_features(X)
        scaler = getattr(self, "scaler_", None)
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

        distances = np.linalg.norm(X_scaled[:, np.newaxis] - self.centroids_, axis=2)
        scores = -distances
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-12)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return cluster labels"""
        self.fit(X)
        return self.labels_
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get detailed information about clusters"""
        if self.centroids_ is None:
            raise ValueError("Model not fitted yet")
        
        cluster_info = {
            'n_clusters': self.n_clusters,
            'iterations': self.iterations_,
            'inertia': self.inertia_,
            'silhouette_score': self.silhouette_,
            'weights': self.weights.tolist(),
            'feature_importance': self.feature_importance.tolist(),
            'centroids': self.centroids_.tolist(),
            'feature_indices': (
                self.feature_indices_.tolist()
                if getattr(self, "feature_indices_", None) is not None
                else None
            ),
            'n_input_features': self.n_input_features_,
            'n_selected_features': self.n_selected_features_,
        }
        
        return cluster_info
    
    def get_cluster_statistics(self, X: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Get statistics for each cluster"""
        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
        
        cluster_stats = {}
        
        for cluster_idx in range(self.n_clusters):
            cluster_points = X[self.labels_ == cluster_idx]
            
            if len(cluster_points) > 0:
                stats = {
                    'count': len(cluster_points),
                    'mean': np.mean(cluster_points, axis=0).tolist(),
                    'std': np.std(cluster_points, axis=0).tolist(),
                    'min': np.min(cluster_points, axis=0).tolist(),
                    'max': np.max(cluster_points, axis=0).tolist(),
                    'weight': self.weights[cluster_idx]
                }
            else:
                stats = {
                    'count': 0,
                    'mean': [0.0] * X.shape[1],
                    'std': [0.0] * X.shape[1],
                    'min': [0.0] * X.shape[1],
                    'max': [0.0] * X.shape[1],
                    'weight': self.weights[cluster_idx]
                }
            
            cluster_stats[cluster_idx] = stats
        
        return cluster_stats
    
    def get_feature_importance_ranking(self) -> List[Tuple[int, float]]:
        """Get feature importance ranking"""
        if self.feature_importance is None:
            raise ValueError("Model not fitted yet")
        
        ranking = sorted(
            enumerate(self.feature_importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranking
    
    def save(self, path: str) -> None:
        """Save model to file"""
        import pickle
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'AdaptiveWeightedClustering':
        """Load model from file"""
        import pickle
        
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        return model


# Utility functions

def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 5,
    n_clusters: int = 3,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for testing"""
    from sklearn.datasets import make_blobs
    
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=random_state
    )
    
    return X, y


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Evaluate clustering quality"""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    evaluation = {}
    
    # Silhouette score
    if len(np.unique(labels)) > 1:
        from sklearn.metrics import silhouette_score
        evaluation['silhouette'] = silhouette_score(X, labels)
    else:
        evaluation['silhouette'] = 0.0
    
    # Davies-Bouldin index
    from sklearn.metrics import davies_bouldin_score
    evaluation['davies_bouldin'] = davies_bouldin_score(X, labels)
    
    # Calinski-Harabasz index
    from sklearn.metrics import calinski_harabasz_score
    evaluation['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    
    # If true labels are provided
    if true_labels is not None:
        evaluation['adjusted_rand'] = adjusted_rand_score(true_labels, labels)
        evaluation['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, labels)
    
    return evaluation


def visualize_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    title: str = "Cluster Visualization"
) -> None:
    """Visualize clusters (2D or 3D)"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    n_features = X.shape[1]
    
    if n_features == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
        
        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X')
        
        plt.title(title)
        plt.xlabel(feature_names[0] if feature_names else 'Feature 1')
        plt.ylabel(feature_names[1] if feature_names else 'Feature 2')
        plt.colorbar(scatter)
        plt.show()
    
    elif n_features == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', alpha=0.6)
        
        if centroids is not None:
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=200, marker='X')
        
        ax.set_title(title)
        ax.set_xlabel(feature_names[0] if feature_names else 'Feature 1')
        ax.set_ylabel(feature_names[1] if feature_names else 'Feature 2')
        ax.set_zlabel(feature_names[2] if feature_names else 'Feature 3')
        fig.colorbar(scatter)
        plt.show()
    
    else:
        print("Visualization only supported for 2D and 3D data")


# Main execution for testing
if __name__ == "__main__":
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=1000, n_features=5, n_clusters=3, random_state=42)
    
    # Create and fit AWC model
    awc = AdaptiveWeightedClustering(n_clusters=3, max_iter=100, random_state=42)
    awc.fit(X)
    
    # Get cluster information
    cluster_info = awc.get_cluster_info()
    print("Cluster Information:")
    print(cluster_info)
    
    # Evaluate clustering
    evaluation = evaluate_clustering(X, awc.labels_)
    print("\nClustering Evaluation:")
    for metric, value in evaluation.items():
        print(f"{metric}: {value:.3f}")
    
    # Visualize clusters (first 3 features)
    visualize_clusters(X[:, :3], awc.labels_, awc.centroids_[:, :3], title="AWC Clusters")
