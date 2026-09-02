"""Tests for ml.clustering module."""

from __future__ import annotations

import numpy as np
import pytest

from ml.clustering import FinancialClusterer, optimal_k

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_well_separated_clusters(
    n_per_cluster: int = 30,
    n_features: int = 4,
    seed: int = 42,
) -> tuple[list[list[float]], list[int]]:
    """Generate 3 well-separated gaussian clusters."""
    rng = np.random.RandomState(seed)
    centers = [
        [10.0] * n_features,
        [-10.0] * n_features,
        [10.0 * ((-1) ** i) for i in range(n_features)],
    ]
    data: list[list[float]] = []
    labels: list[int] = []
    for cluster_id, center in enumerate(centers):
        pts = rng.randn(n_per_cluster, n_features) * 0.5 + np.array(center)
        data.extend(pts.tolist())
        labels.extend([cluster_id] * n_per_cluster)
    return data, labels


def _make_with_outliers(seed: int = 42) -> list[list[float]]:
    """Generate tight cluster + distant outliers."""
    rng = np.random.RandomState(seed)
    core = (rng.randn(50, 3) * 0.3).tolist()
    outliers = [[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]]
    return core + outliers


# ---------------------------------------------------------------------------
# KMeans basics
# ---------------------------------------------------------------------------


class TestKMeansClustering:
    def test_fit_creates_labels(self):
        data, _ = _make_well_separated_clusters()
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data)
        assert c.labels_ is not None
        assert len(c.labels_) == len(data)

    def test_correct_number_of_clusters(self):
        data, _ = _make_well_separated_clusters()
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data)
        assert len(set(c.labels_)) == 3

    def test_cluster_centers_exist(self):
        data, _ = _make_well_separated_clusters()
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data)
        assert c.cluster_centers_ is not None
        assert c.cluster_centers_.shape[0] == 3

    def test_labels_recover_structure(self):
        """Each true cluster should map predominantly to one predicted cluster."""
        data, true_labels = _make_well_separated_clusters()
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data)
        # Check that predicted labels partition similarly to true labels
        # (ignoring label permutation)
        from collections import Counter

        for true_id in range(3):
            indices = [i for i, tl in enumerate(true_labels) if tl == true_id]
            pred = [int(c.labels_[i]) for i in indices]
            most_common = Counter(pred).most_common(1)[0][1]
            # At least 90% should share the same predicted label
            assert most_common / len(indices) >= 0.9

    def test_scaler_and_pca_stored(self):
        data, _ = _make_well_separated_clusters()
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data)
        assert c.scaler_ is not None
        assert c.pca_ is not None
        assert c.pca_coords_ is not None
        assert c.pca_coords_.shape == (len(data), 2)


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------


class TestDBSCAN:
    def test_dbscan_fit(self):
        data, _ = _make_well_separated_clusters(n_per_cluster=40)
        c = FinancialClusterer(method="dbscan")
        c.fit(data)
        assert c.labels_ is not None
        assert len(c.labels_) == len(data)

    def test_dbscan_no_cluster_centers(self):
        data, _ = _make_well_separated_clusters(n_per_cluster=40)
        c = FinancialClusterer(method="dbscan")
        c.fit(data)
        assert c.cluster_centers_ is None

    def test_dbscan_predict_fallback(self):
        data, _ = _make_well_separated_clusters(n_per_cluster=40)
        c = FinancialClusterer(method="dbscan")
        c.fit(data)
        preds = c.predict([data[0]])
        assert len(preds) == 1
        assert isinstance(preds[0], int)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_after_fit(self):
        data, _ = _make_well_separated_clusters()
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data)
        preds = c.predict(data[:5])
        assert len(preds) == 5
        for p in preds:
            assert isinstance(p, int)
            assert 0 <= p < 3

    def test_predict_new_data_near_cluster(self):
        data, _ = _make_well_separated_clusters()
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data)
        # Point near the first cluster center [10, 10, 10, 10]
        near_first = [[10.1, 10.1, 10.1, 10.1]]
        pred = c.predict(near_first)
        # Should match the cluster assigned to other points near [10, 10, 10, 10]
        ref_pred = c.predict([[10.0, 10.0, 10.0, 10.0]])
        assert pred == ref_pred

    def test_predict_raises_before_fit(self):
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        with pytest.raises(RuntimeError, match="not fitted"):
            c.predict([[1.0, 2.0]])


# ---------------------------------------------------------------------------
# Cluster profiles
# ---------------------------------------------------------------------------


class TestClusterProfiles:
    def test_returns_one_profile_per_cluster(self):
        data, _ = _make_well_separated_clusters()
        names = [f"f{i}" for i in range(4)]
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data, feature_names=names)
        profiles = c.get_cluster_profiles(data, names)
        assert len(profiles) == 3

    def test_profile_structure(self):
        data, _ = _make_well_separated_clusters()
        names = [f"f{i}" for i in range(4)]
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data, feature_names=names)
        profiles = c.get_cluster_profiles(data, names)
        for p in profiles:
            assert "cluster_id" in p
            assert "size" in p
            assert "mean_features" in p
            assert "distinguishing_features" in p
            assert isinstance(p["mean_features"], dict)
            assert p["size"] > 0

    def test_profile_sizes_sum_to_total(self):
        data, _ = _make_well_separated_clusters()
        names = [f"f{i}" for i in range(4)]
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data, feature_names=names)
        profiles = c.get_cluster_profiles(data, names)
        total = sum(p["size"] for p in profiles)
        assert total == len(data)

    def test_raises_before_fit(self):
        c = FinancialClusterer()
        with pytest.raises(RuntimeError):
            c.get_cluster_profiles([[1.0]], ["f0"])


# ---------------------------------------------------------------------------
# Find peers
# ---------------------------------------------------------------------------


class TestFindPeers:
    def test_returns_k_results(self):
        data, _ = _make_well_separated_clusters()
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data)
        peers = c.find_peers(data[0], k=5)
        assert len(peers) == 5

    def test_sorted_by_distance(self):
        data, _ = _make_well_separated_clusters()
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data)
        peers = c.find_peers(data[0], k=5)
        distances = [d for _, d in peers]
        assert distances == sorted(distances)

    def test_returns_index_distance_tuples(self):
        data, _ = _make_well_separated_clusters()
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data)
        peers = c.find_peers(data[0], k=3)
        for idx, dist in peers:
            assert isinstance(idx, int)
            assert isinstance(dist, float)
            assert dist >= 0.0

    def test_k_greater_than_n_samples(self):
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        c = FinancialClusterer(n_clusters=2, method="kmeans")
        c.fit(data)
        peers = c.find_peers(data[0], k=100)
        assert len(peers) == 3  # only 3 samples available

    def test_raises_before_fit(self):
        c = FinancialClusterer()
        with pytest.raises(RuntimeError):
            c.find_peers([1.0, 2.0])


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


class TestOutliers:
    def test_detects_planted_outliers(self):
        data = _make_with_outliers()
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data)
        outliers = c.get_outliers(data)
        # The last two points (indices 50, 51) are extreme outliers
        assert 50 in outliers or 51 in outliers

    def test_returns_list_of_ints(self):
        data = _make_with_outliers()
        c = FinancialClusterer()
        outliers = c.get_outliers(data)
        assert isinstance(outliers, list)
        for idx in outliers:
            assert isinstance(idx, int)

    def test_single_sample_returns_empty(self):
        c = FinancialClusterer()
        outliers = c.get_outliers([[1.0, 2.0]])
        assert outliers == []


# ---------------------------------------------------------------------------
# Visualization data
# ---------------------------------------------------------------------------


class TestVisualizationData:
    def test_returns_2d_coords(self):
        data, _ = _make_well_separated_clusters()
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data)
        viz = c.get_visualization_data()
        assert "pca_coords" in viz
        assert "labels" in viz
        assert len(viz["pca_coords"]) == len(data)
        for point in viz["pca_coords"]:
            assert len(point) == 2

    def test_cluster_centers_2d_for_kmeans(self):
        data, _ = _make_well_separated_clusters()
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data)
        viz = c.get_visualization_data()
        assert viz["cluster_centers_2d"] is not None
        assert len(viz["cluster_centers_2d"]) == 3
        for pt in viz["cluster_centers_2d"]:
            assert len(pt) == 2

    def test_cluster_centers_none_for_dbscan(self):
        data, _ = _make_well_separated_clusters(n_per_cluster=40)
        c = FinancialClusterer(method="dbscan")
        c.fit(data)
        viz = c.get_visualization_data()
        assert viz["cluster_centers_2d"] is None

    def test_raises_before_fit(self):
        c = FinancialClusterer()
        with pytest.raises(RuntimeError):
            c.get_visualization_data()


# ---------------------------------------------------------------------------
# optimal_k
# ---------------------------------------------------------------------------


class TestOptimalK:
    def test_returns_reasonable_k_for_3_clusters(self):
        data, _ = _make_well_separated_clusters(n_per_cluster=50)
        k = optimal_k(data, max_k=8)
        # Well-separated clusters: should suggest 2-4
        assert 2 <= k <= 5

    def test_returns_int(self):
        data, _ = _make_well_separated_clusters()
        k = optimal_k(data)
        assert isinstance(k, int)

    def test_min_k_is_2(self):
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        k = optimal_k(data, max_k=10)
        assert k >= 2

    def test_small_dataset(self):
        data = [[1.0], [2.0], [3.0]]
        k = optimal_k(data, max_k=10)
        assert k == 2  # only 2 possible with 3 samples (upper = n-1 = 2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_sample(self):
        c = FinancialClusterer(n_clusters=1, method="kmeans")
        c.fit([[1.0, 2.0, 3.0]])
        assert len(c.labels_) == 1

    def test_two_samples(self):
        c = FinancialClusterer(n_clusters=2, method="kmeans")
        c.fit([[1.0, 2.0], [10.0, 20.0]])
        assert len(c.labels_) == 2
        assert len(set(c.labels_)) == 2

    def test_all_identical_features(self):
        data = [[5.0, 5.0, 5.0]] * 10
        c = FinancialClusterer(n_clusters=2, method="kmeans")
        c.fit(data)
        # Should still produce labels without error
        assert len(c.labels_) == 10

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unsupported method"):
            FinancialClusterer(method="spectral")

    def test_single_feature(self):
        data = [[float(i)] for i in range(20)]
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data)
        assert c.pca_coords_.shape == (20, 2)

    def test_optimal_k_two_samples(self):
        k = optimal_k([[1.0], [2.0]], max_k=10)
        assert k == 2

    def test_find_peers_identical_points(self):
        data = [[1.0, 1.0]] * 5
        c = FinancialClusterer(n_clusters=1, method="kmeans")
        c.fit(data)
        peers = c.find_peers([1.0, 1.0], k=3)
        assert len(peers) == 3
        for _, dist in peers:
            assert dist == pytest.approx(0.0, abs=1e-6)

    def test_feature_names_stored(self):
        data, _ = _make_well_separated_clusters()
        names = [f"feat_{i}" for i in range(4)]
        c = FinancialClusterer(n_clusters=3, method="kmeans")
        c.fit(data, feature_names=names)
        assert c._feature_names == names
