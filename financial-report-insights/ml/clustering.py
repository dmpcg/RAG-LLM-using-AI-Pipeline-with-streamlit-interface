"""Clustering and peer analysis for financial data.

Provides unsupervised clustering (KMeans, DBSCAN), peer-company identification,
outlier detection, and visualization-ready output using scikit-learn.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class FinancialClusterer:
    """Cluster financial entities and find peers.

    Parameters
    ----------
    n_clusters : int
        Number of clusters for KMeans (ignored for DBSCAN).
    method : str
        Clustering algorithm: ``"kmeans"`` or ``"dbscan"``.
    """

    def __init__(self, n_clusters: int = 5, method: str = "kmeans") -> None:
        if method not in ("kmeans", "dbscan"):
            raise ValueError(f"Unsupported method '{method}'. Use 'kmeans' or 'dbscan'.")
        self.n_clusters = n_clusters
        self.method = method

        # Fitted state
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.pca_coords_: Optional[np.ndarray] = None
        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[PCA] = None
        self._model: object = None
        self._scaled_matrix: Optional[np.ndarray] = None
        self._feature_names: Optional[list[str]] = None

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(
        self,
        feature_matrix: list[list[float]],
        feature_names: Optional[list[str]] = None,
    ) -> None:
        """Fit clustering model on *feature_matrix*.

        Scales features, reduces to 2D via PCA for visualization, and runs
        the selected clustering algorithm.
        """
        if len(feature_matrix) == 0:
            raise ValueError("Cannot fit clusterer on empty feature matrix")
        matrix = np.asarray(feature_matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {matrix.shape}")
        n_samples, n_features = matrix.shape

        self._feature_names = feature_names

        # Scale
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(matrix)
        self._scaled_matrix = scaled

        # PCA to 2D (handle edge cases)
        n_components = min(2, n_samples, n_features)
        self.pca_ = PCA(n_components=n_components)
        self.pca_coords_ = self.pca_.fit_transform(scaled)
        # Pad to 2 columns if only 1 component
        if self.pca_coords_.shape[1] < 2:
            self.pca_coords_ = np.column_stack([self.pca_coords_, np.zeros(n_samples)])

        # Cluster
        if self.method == "kmeans":
            effective_k = min(self.n_clusters, n_samples)
            model = KMeans(
                n_clusters=effective_k,
                n_init="auto",
                random_state=42,
            )
            model.fit(scaled)
            self.labels_ = model.labels_
            self.cluster_centers_ = model.cluster_centers_
            self._model = model
        else:
            model = DBSCAN(eps=0.5, min_samples=max(2, n_samples // 10))
            model.fit(scaled)
            self.labels_ = model.labels_
            self.cluster_centers_ = None
            self._model = model

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #

    def predict(self, features: list[list[float]]) -> list[int]:
        """Assign new samples to existing clusters.

        For DBSCAN (which has no native predict), assigns each sample to
        the nearest cluster centroid computed from the fitted data, or -1
        if the model produced no clusters.
        """
        if self.scaler_ is None or self.labels_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        arr = np.array(features, dtype=np.float64)
        scaled = self.scaler_.transform(arr)

        if self.method == "kmeans":
            return [int(x) for x in self._model.predict(scaled)]  # type: ignore[union-attr]

        # DBSCAN fallback: nearest centroid
        unique_labels = set(self.labels_) - {-1}
        if not unique_labels:
            return [-1] * len(features)

        centroids = {}
        for label in unique_labels:
            mask = self.labels_ == label
            centroids[label] = self._scaled_matrix[mask].mean(axis=0)

        results: list[int] = []
        for row in scaled:
            best_label = -1
            best_dist = float("inf")
            for label, centroid in centroids.items():
                dist = float(np.linalg.norm(row - centroid))
                if dist < best_dist:
                    best_dist = dist
                    best_label = int(label)
            results.append(best_label)
        return results

    # ------------------------------------------------------------------ #
    # Cluster profiles
    # ------------------------------------------------------------------ #

    def get_cluster_profiles(
        self,
        feature_matrix: list[list[float]],
        feature_names: list[str],
    ) -> list[dict]:
        """Return a profile dict for each cluster.

        Each profile contains:
        * ``cluster_id`` -- integer label
        * ``size`` -- number of samples
        * ``mean_features`` -- dict of feature name to mean value
        * ``distinguishing_features`` -- top features that differ most
          from the overall mean
        """
        if self.labels_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        matrix = np.array(feature_matrix, dtype=np.float64)
        overall_mean = matrix.mean(axis=0)

        unique_labels = sorted(set(self.labels_))
        profiles: list[dict] = []

        for label in unique_labels:
            mask = self.labels_ == label
            cluster_data = matrix[mask]
            cluster_mean = cluster_data.mean(axis=0)

            mean_features = {name: float(cluster_mean[i]) for i, name in enumerate(feature_names)}

            # Distinguishing features: largest absolute deviation from overall mean
            deviations = np.abs(cluster_mean - overall_mean)
            # Normalise by overall std to get relative importance
            overall_std = matrix.std(axis=0)
            safe_std = np.where(overall_std < 1e-12, 1.0, overall_std)
            normed_dev = deviations / safe_std
            top_indices = np.argsort(normed_dev)[::-1][:5]

            distinguishing = [
                {
                    "feature": feature_names[i],
                    "cluster_mean": float(cluster_mean[i]),
                    "overall_mean": float(overall_mean[i]),
                    "deviation": float(normed_dev[i]),
                }
                for i in top_indices
            ]

            profiles.append(
                {
                    "cluster_id": int(label),
                    "size": int(mask.sum()),
                    "mean_features": mean_features,
                    "distinguishing_features": distinguishing,
                }
            )

        return profiles

    # ------------------------------------------------------------------ #
    # Peer finding
    # ------------------------------------------------------------------ #

    def find_peers(
        self,
        features: list[float],
        k: int = 5,
    ) -> list[tuple[int, float]]:
        """Find *k* nearest neighbors in scaled feature space.

        Returns list of ``(index, distance)`` sorted by distance ascending.
        """
        if self.scaler_ is None or self._scaled_matrix is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_samples = self._scaled_matrix.shape[0]
        effective_k = min(k, n_samples)

        arr = np.array(features, dtype=np.float64).reshape(1, -1)
        scaled = self.scaler_.transform(arr)

        nn = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
        nn.fit(self._scaled_matrix)
        distances, indices = nn.kneighbors(scaled)

        return [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]

    # ------------------------------------------------------------------ #
    # Outlier detection
    # ------------------------------------------------------------------ #

    def get_outliers(self, feature_matrix: list[list[float]]) -> list[int]:
        """Detect outlier samples using Isolation Forest.

        Returns indices of samples flagged as outliers.
        """
        matrix = np.array(feature_matrix, dtype=np.float64)
        n_samples = matrix.shape[0]

        if n_samples < 2:
            return []

        iso = IsolationForest(
            contamination="auto",
            random_state=42,
            n_estimators=100,
        )
        preds = iso.fit_predict(matrix)
        return [int(i) for i, p in enumerate(preds) if p == -1]

    # ------------------------------------------------------------------ #
    # Visualization data
    # ------------------------------------------------------------------ #

    def get_visualization_data(self) -> dict:
        """Return PCA 2D coordinates, labels, and projected cluster centers.

        Designed for front-end plotting without matplotlib.
        """
        if self.pca_coords_ is None or self.labels_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        result: dict = {
            "pca_coords": self.pca_coords_.tolist(),
            "labels": [int(lb) for lb in self.labels_],
        }

        if self.cluster_centers_ is not None and self.pca_ is not None:
            projected = self.pca_.transform(self.cluster_centers_)
            if projected.shape[1] < 2:
                projected = np.column_stack([projected, np.zeros(projected.shape[0])])
            result["cluster_centers_2d"] = projected.tolist()
        else:
            result["cluster_centers_2d"] = None

        return result


# ===================================================================
# Standalone utility
# ===================================================================


def optimal_k(
    feature_matrix: list[list[float]],
    max_k: int = 10,
) -> int:
    """Recommend number of clusters using elbow + silhouette heuristics.

    Parameters
    ----------
    feature_matrix : list of feature vectors
    max_k : maximum k to evaluate

    Returns
    -------
    Recommended k (>= 2).
    """
    matrix = np.array(feature_matrix, dtype=np.float64)
    n_samples = matrix.shape[0]

    if n_samples < 3:
        return min(n_samples, 2)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    upper = min(max_k, n_samples - 1)
    if upper < 2:
        return 2

    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in range(2, upper + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        km.fit(scaled)
        inertias.append(float(km.inertia_))
        if len(set(km.labels_)) > 1:
            silhouettes.append(float(silhouette_score(scaled, km.labels_)))
        else:
            silhouettes.append(-1.0)

    # Best silhouette score
    best_sil_idx = int(np.argmax(silhouettes))

    # Elbow detection: largest drop in inertia
    if len(inertias) >= 3:
        diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        second_diffs = [diffs[i] - diffs[i + 1] for i in range(len(diffs) - 1)]
        elbow_idx = int(np.argmax(second_diffs))
    else:
        elbow_idx = 0

    # Prefer silhouette; fall back to elbow if silhouette is poor
    if silhouettes[best_sil_idx] > 0.3:
        return best_sil_idx + 2
    return elbow_idx + 2
