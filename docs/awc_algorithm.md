# AWC (Adaptive Weight Clustering) Algorithm Documentation

## Overview

Adaptive Weight Clustering (AWC) is a novel clustering algorithm specifically designed for duck egg fertility detection. It extends traditional clustering approaches by dynamically adapting cluster weights based on feature importance and cluster quality metrics.

## Algorithm Architecture

### Core Components

1. **Feature Extraction & Normalization**
2. **Initial Centroid Initialization (K-Means++)**
3. **Iterative Weight Adaptation**
4. **Centroid Update with Weighted Points**
5. **Feature Importance Calculation**
6. **Convergence Check**

## Mathematical Foundation

### 1. Feature Normalization

All input features are standardized using StandardScaler:

```
X_scaled = (X - μ) / σ
```

Where:
- μ = mean of each feature
- σ = standard deviation of each feature

### 2. Distance Calculation

For each sample x_i and cluster centroid c_j:

```
d(x_i, c_j) = ||x_i - c_j||²
```

### 3. Cluster Assignment

Each sample is assigned to the nearest centroid:

```
label(x_i) = argmin_j(d(x_i, c_j))
```

### 4. Weight Calculation

The adaptive weight for cluster k is calculated based on intra-cluster and inter-cluster distances:

```
weight_k = (min_inter_cluster_distance_k) / (intra_cluster_distance_k)
```

Where:
- **intra_cluster_distance_k**: Average distance of points in cluster k to its centroid
- **min_inter_cluster_distance_k**: Minimum distance between centroid k and all other centroids

Weights are normalized:

```
weight_k = weight_k / Σ(weight_i) for all i
```

### 5. Centroid Update

Centroids are updated using weighted average:

```
c_k = (1/|C_k|) * Σ(x_i) for all x_i ∈ C_k
```

Where C_k is the set of points in cluster k.

### 6. Feature Importance

Feature importance is calculated based on cluster separation:

```
importance_f = Σ_k Σ_j≠k (|μ_k,f - μ_j,f| / (σ²_k,f + ε))
```

Where:
- μ_k,f = mean of feature f in cluster k
- σ²_k,f = variance of feature f in cluster k
- ε = small constant (1e-6) for numerical stability

Importance values are clipped to [0.1, 10.0] and normalized.

### 7. Inertia (Loss Function)

```
Inertia = Σ_i min_j(d(x_i, c_j))
```

## Algorithm Steps

### Training Phase

```
1. Input: Dataset X with n samples and m features
2. Normalize X → X_scaled
3. Initialize centroids using K-Means++
4. For iteration = 1 to max_iterations:
   a. Assign each point to nearest centroid
   b. Calculate new centroids from assigned points
   c. Calculate adaptive weights for each cluster
   d. Update feature importance scores
   e. Calculate inertia
   f. If |inertia_prev - inertia| < tolerance:
      - Break (converged)
5. Store final labels, centroids, weights
```

### Prediction Phase

```
1. Input: New data point x
2. Normalize x using training scaler
3. Calculate distances to all centroids
4. Assign to nearest centroid
5. Return cluster label
```

## Key Features

### Adaptive Weighting
- Clusters with better separation (high inter-cluster, low intra-cluster distance) receive higher weights
- Automatically adjusts to data distribution

### Feature Selection
- Identifies most discriminative features for classification
- Can be used for dimensionality reduction

### Robust Initialization
- K-Means++ initialization prevents poor local minima
- Multiple random restarts available

### Convergence Criteria
- Monitors inertia changes
- Stops when improvement < tolerance
- Maximum iteration limit prevents infinite loops

## Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_clusters | int | 3 | Number of clusters (fertile, infertile, uncertain) |
| max_iter | int | 100 | Maximum iterations |
| tol | float | 1e-4 | Convergence tolerance |
| initial_weights | list | None | Initial cluster weights |
| feature_importance | list | None | Initial feature importance |
| random_state | int | None | Random seed |

## Implementation Details

### Class: AdaptiveWeightedClustering

**Key Methods:**

- `fit(X)`: Train the model
- `predict(X)`: Predict cluster labels
- `predict_proba(X)`: Return cluster probabilities
- `fit_predict(X)`: Fit and predict in one step
- `get_cluster_info()`: Detailed cluster statistics
- `get_feature_importance_ranking()`: Rank features by importance
- `save(path)`: Save model to file
- `load(path)`: Load model from file

**Attributes:**

- `centroids_`: Cluster centers
- `labels_`: Training data labels
- `inertia_`: Sum of squared distances
- `silhouette_`: Silhouette score
- `weights`: Cluster weights
- `feature_importance`: Feature importance scores
- `scaler_`: Fitted StandardScaler

## Evaluation Metrics

### Internal Metrics

1. **Silhouette Score**: [-1, 1], higher is better
   - Measures cluster cohesion and separation
   
2. **Davies-Bouldin Index**: Lower is better
   - Average similarity between clusters
   
3. **Calinski-Harabasz Index**: Higher is better
   - Ratio of between-cluster to within-cluster dispersion

### External Metrics (if true labels available)

1. **Adjusted Rand Index**: [-1, 1], higher is better
2. **Normalized Mutual Information**: [0, 1], higher is better

## Use Cases in Duck Egg Fertility

### Cluster Interpretation

- **Cluster 0**: Fertile eggs (developing embryo)
- **Cluster 1**: Infertile eggs (no development)
- **Cluster 2**: Uncertain/transition cases

### Feature Types

1. **Classical Features**:
   - GLCM: Contrast, correlation, energy, homogeneity
   - LBP: Texture patterns
   - Morphological: Area, perimeter, circularity

2. **Deep Features**:
   - U-Net bottleneck embeddings
   - Segmentation mask statistics

### Decision Thresholds

- Fertile: Cluster with highest weight
- Infertile: Cluster with lowest weight
- Uncertain: Middle cluster or low confidence

## Advantages

1. **Adaptive**: Automatically adjusts to data characteristics
2. **Interpretable**: Feature importance provides insights
3. **Robust**: Handles noise and outliers
4. **Flexible**: Works with various feature types
5. **Scalable**: Efficient for large datasets

## Limitations

1. Requires careful initialization
2. Sensitive to feature scaling
3. Assumes spherical clusters
4. Number of clusters must be specified
5. May converge to local optima

## Performance Optimization

### Speed Improvements

1. Use sparse matrices for high-dimensional features
2. Implement mini-batch updates
3. Parallel distance calculations
4. Early stopping on convergence

### Quality Improvements

1. Ensemble multiple runs
2. Combine with other clustering methods
3. Use domain knowledge for initialization
4. Feature engineering and selection

## References

1. Khairanmarzuki et al. "Adaptive Weighted Clustering for Egg Fertility Detection"
2. Arthur, D. & Vassilvitskii, S. (2007). K-Means++: The Advantages of Careful Seeding
3. Rousseeuw, P.J. (1987). Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis

---

*Last updated: 2026-04-27*
