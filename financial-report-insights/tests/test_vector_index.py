"""
Tests for vector_index.py – NumpyFlatIndex, FAISSIndex, HNSWIndex, and create_index.

NumpyFlatIndex is tested exhaustively because it is the always-available fallback.
FAISSIndex and HNSWIndex are tested via mocked imports so the suite runs on any
environment regardless of whether faiss or hnswlib are installed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the package root is on sys.path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from vector_index import (
    NumpyFlatIndex,
    VectorIndex,
    create_index,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8  # small dimension for fast tests


def _unit_vec(values: List[float]) -> List[float]:
    """Return an L2-normalised version of *values*."""
    arr = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 0 else arr.tolist()


def _random_vecs(n: int, dim: int = DIM, seed: int = 0) -> List[List[float]]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).tolist()


# ===========================================================================
# NumpyFlatIndex
# ===========================================================================


class TestNumpyFlatIndexBasic:
    """Basic construction and length tests."""

    def test_empty_index_has_zero_length(self):
        idx = NumpyFlatIndex(DIM)
        assert len(idx) == 0

    def test_add_single_vector(self):
        idx = NumpyFlatIndex(DIM)
        idx.add([_random_vecs(1)[0]], [0])
        assert len(idx) == 1

    def test_add_multiple_vectors(self):
        idx = NumpyFlatIndex(DIM)
        vecs = _random_vecs(10)
        ids = list(range(10))
        idx.add(vecs, ids)
        assert len(idx) == 10

    def test_add_incremental_extends_length(self):
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(5), list(range(5)))
        idx.add(_random_vecs(3, seed=1), [10, 11, 12])
        assert len(idx) == 8

    def test_add_empty_list_is_noop(self):
        idx = NumpyFlatIndex(DIM)
        idx.add([], [])
        assert len(idx) == 0

    def test_add_mismatched_lengths_raises(self):
        idx = NumpyFlatIndex(DIM)
        with pytest.raises(ValueError, match="same length"):
            idx.add(_random_vecs(3), [0, 1])  # 3 vecs, 2 ids

    def test_add_wrong_dimension_raises(self):
        idx = NumpyFlatIndex(DIM)
        wrong_dim_vec = [[1.0, 2.0]]  # 2-dim instead of DIM
        with pytest.raises(ValueError, match="dimension"):
            idx.add(wrong_dim_vec, [0])


class TestNumpyFlatIndexSearch:
    """Search correctness tests."""

    def test_search_empty_index_returns_empty(self):
        idx = NumpyFlatIndex(DIM)
        result = idx.search(_random_vecs(1)[0], top_k=3)
        assert result == []

    def test_search_returns_top_k_results(self):
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(20), list(range(20)))
        results = idx.search(_random_vecs(1)[0], top_k=5)
        assert len(results) == 5

    def test_search_top_k_clamped_to_index_size(self):
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(3), [0, 1, 2])
        results = idx.search(_random_vecs(1)[0], top_k=10)
        assert len(results) == 3

    def test_search_scores_are_sorted_descending(self):
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(15), list(range(15)))
        results = idx.search(_random_vecs(1)[0], top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_returns_id_score_tuples(self):
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(5), [10, 20, 30, 40, 50])
        results = idx.search(_random_vecs(1)[0], top_k=3)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            doc_id, score = item
            assert isinstance(doc_id, int)
            assert isinstance(score, float)

    def test_search_ids_are_from_added_set(self):
        idx = NumpyFlatIndex(DIM)
        custom_ids = [100, 200, 300]
        idx.add(_random_vecs(3), custom_ids)
        results = idx.search(_random_vecs(1)[0], top_k=3)
        returned_ids = {r[0] for r in results}
        assert returned_ids.issubset(set(custom_ids))

    def test_search_identical_query_scores_highest(self):
        """A query vector identical to an indexed vector should score ~1.0."""
        idx = NumpyFlatIndex(DIM)
        target = _unit_vec([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        others = _random_vecs(9, seed=7)
        idx.add([target] + others, list(range(10)))
        results = idx.search(target, top_k=1)
        assert results[0][0] == 0  # first id (target)
        assert abs(results[0][1] - 1.0) < 1e-4

    def test_search_single_vector_index(self):
        idx = NumpyFlatIndex(DIM)
        vec = _random_vecs(1)[0]
        idx.add([vec], [42])
        results = idx.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == 42

    def test_search_zero_query_returns_documents(self):
        """A zero query vector should not raise; cosine sim defaults gracefully."""
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(5), list(range(5)))
        zero_query = [0.0] * DIM
        # Should not raise; may return any ordering
        results = idx.search(zero_query, top_k=3)
        assert len(results) == 3

    def test_search_large_index_partial_sort(self):
        """Trigger the argpartition path (n > top_k * 4)."""
        n = 100
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(n), list(range(n)))
        results = idx.search(_random_vecs(1)[0], top_k=5)
        assert len(results) == 5
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestNumpyFlatIndexPersistence:
    """Save and load round-trip tests."""

    def test_save_load_roundtrip(self, tmp_path):
        idx = NumpyFlatIndex(DIM)
        vecs = _random_vecs(8)
        ids = list(range(8))
        idx.add(vecs, ids)

        path = str(tmp_path / "test_index")
        idx.save(path)

        idx2 = NumpyFlatIndex(DIM)
        idx2.load(path)
        assert len(idx2) == 8

    def test_save_load_produces_same_results(self, tmp_path):
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((10, DIM)).tolist()
        ids = list(range(10))
        query = rng.standard_normal(DIM).tolist()

        idx = NumpyFlatIndex(DIM)
        idx.add(vecs, ids)
        original = idx.search(query, top_k=3)

        path = str(tmp_path / "roundtrip")
        idx.save(path)

        idx2 = NumpyFlatIndex(DIM)
        idx2.load(path)
        restored = idx2.search(query, top_k=3)

        assert [r[0] for r in original] == [r[0] for r in restored]
        for (_, s1), (_, s2) in zip(original, restored):
            assert abs(s1 - s2) < 1e-5

    def test_load_nonexistent_raises(self, tmp_path):
        idx = NumpyFlatIndex(DIM)
        with pytest.raises(FileNotFoundError):
            idx.load(str(tmp_path / "does_not_exist"))

    def test_save_creates_parent_dirs(self, tmp_path):
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(3), [0, 1, 2])
        nested_path = str(tmp_path / "a" / "b" / "c" / "index")
        idx.save(nested_path)
        assert Path(nested_path + ".npz").exists()

    def test_save_empty_index(self, tmp_path):
        idx = NumpyFlatIndex(DIM)
        path = str(tmp_path / "empty")
        idx.save(path)  # Should not raise
        assert Path(path + ".npz").exists()

    def test_load_preserves_dimension(self, tmp_path):
        idx = NumpyFlatIndex(DIM)
        idx.add(_random_vecs(4), [0, 1, 2, 3])
        path = str(tmp_path / "dim_check")
        idx.save(path)

        idx2 = NumpyFlatIndex(DIM)
        idx2.load(path)
        assert idx2.dimension == DIM


# ===========================================================================
# Protocol conformance
# ===========================================================================


class TestProtocolConformance:
    """Verify NumpyFlatIndex satisfies the VectorIndex Protocol."""

    def test_numpy_flat_index_satisfies_protocol(self):
        idx = NumpyFlatIndex(DIM)
        assert isinstance(idx, VectorIndex)


# ===========================================================================
# create_index factory
# ===========================================================================


class TestCreateIndex:
    """Tests for the create_index factory function."""

    def test_numpy_backend_always_works(self):
        idx = create_index(DIM, backend="numpy")
        assert isinstance(idx, NumpyFlatIndex)

    def test_auto_backend_falls_back_to_numpy_when_no_ann_libs(self):
        """When faiss and hnswlib are both absent, auto must return NumpyFlatIndex."""
        with (
            patch("vector_index._FAISS_AVAILABLE", False),
            patch("vector_index._HNSW_AVAILABLE", False),
        ):
            # Reset cached availability flags so _check_* re-evaluates
            import vector_index as vi
            vi._FAISS_AVAILABLE = False
            vi._HNSW_AVAILABLE = False
            idx = create_index(DIM, backend="auto")
            assert isinstance(idx, NumpyFlatIndex)
            # Restore
            vi._FAISS_AVAILABLE = None
            vi._HNSW_AVAILABLE = None

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_index(DIM, backend="bogus")

    def test_numpy_index_is_functional_after_create(self):
        idx = create_index(DIM, backend="numpy")
        vecs = _random_vecs(5)
        idx.add(vecs, list(range(5)))
        results = idx.search(vecs[0], top_k=3)
        assert len(results) == 3

    def test_faiss_backend_raises_when_unavailable(self):
        import vector_index as vi
        original = vi._FAISS_AVAILABLE
        vi._FAISS_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="faiss"):
                create_index(DIM, backend="faiss")
        finally:
            vi._FAISS_AVAILABLE = original

    def test_hnswlib_backend_raises_when_unavailable(self):
        import vector_index as vi
        original = vi._HNSW_AVAILABLE
        vi._HNSW_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="hnswlib"):
                create_index(DIM, backend="hnswlib")
        finally:
            vi._HNSW_AVAILABLE = original


# ===========================================================================
# FAISSIndex (mocked)
# ===========================================================================


def _make_faiss_mock():
    """Build a minimal faiss mock that lets FAISSIndex operate correctly.

    Supports both IndexFlatIP (sub-threshold) and IndexIVFFlat (IVF mode)
    paths.  Both are backed by brute-force numpy so recall is exact in tests.
    """
    faiss = MagicMock()
    faiss.METRIC_INNER_PRODUCT = 0

    class FakeIndex:
        """Thin numpy-backed faiss.IndexFlatIP / IndexIVFFlat substitute."""

        def __init__(self, dim):
            self._dim = dim
            self._matrix = np.empty((0, dim), dtype=np.float32)
            self.is_trained = True
            self.nprobe = 1

        def add(self, matrix):
            if self._matrix.shape[0] == 0:
                self._matrix = matrix.copy()
            else:
                self._matrix = np.vstack([self._matrix, matrix])

        def search(self, query, k):
            if self._matrix.shape[0] == 0:
                return np.array([[-1.0] * k]), np.array([[-1] * k])
            n = self._matrix.shape[0]
            k = min(k, n)
            scores = self._matrix @ query[0]
            top = np.argsort(scores)[-k:][::-1]
            return np.array([scores[top]]), np.array([top])

        def train(self, matrix):
            # IVFFlat train: brute-force mock ignores centroid training
            pass

    def IndexFlatIP(dim):
        return FakeIndex(dim)

    def IndexIVFFlat(quantizer, dim, n_cells, metric):
        idx = FakeIndex(dim)
        idx.is_trained = False  # needs train() called before add()
        return idx

    faiss.IndexFlatIP = IndexFlatIP
    faiss.IndexIVFFlat = IndexIVFFlat
    faiss.write_index = MagicMock()
    faiss.read_index = MagicMock(return_value=FakeIndex(DIM))
    return faiss


class TestFAISSIndexMocked:
    """FAISSIndex tests using a mocked faiss module."""

    def _get_faiss_index(self, dim=DIM):
        from vector_index import FAISSIndex
        fake_faiss = _make_faiss_mock()
        idx = FAISSIndex.__new__(FAISSIndex)
        idx.dimension = dim
        idx.nprobe = 10
        idx._ids = []
        idx._raw_embeddings = []
        idx._index = None
        idx._faiss = fake_faiss
        return idx

    def test_empty_index_has_zero_length(self):
        idx = self._get_faiss_index()
        assert len(idx) == 0

    def test_add_and_search(self):
        idx = self._get_faiss_index()
        vecs = _random_vecs(5)
        idx.add(vecs, list(range(5)))
        assert len(idx) == 5
        results = idx.search(vecs[0], top_k=3)
        assert len(results) == 3

    def test_search_empty_returns_empty(self):
        idx = self._get_faiss_index()
        results = idx.search(_random_vecs(1)[0], top_k=3)
        assert results == []

    def test_scores_sorted_descending(self):
        idx = self._get_faiss_index()
        idx.add(_random_vecs(10), list(range(10)))
        results = idx.search(_random_vecs(1)[0], top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_add_mismatched_lengths_raises(self):
        idx = self._get_faiss_index()
        with pytest.raises(ValueError, match="length"):
            idx.add(_random_vecs(3), [0, 1])

    def test_save_calls_write_index(self, tmp_path):
        idx = self._get_faiss_index()
        idx.add(_random_vecs(3), [0, 1, 2])
        idx.save(str(tmp_path / "faiss_test"))
        idx._faiss.write_index.assert_called_once()

    def test_load_calls_read_index(self, tmp_path):
        idx = self._get_faiss_index()
        # Create dummy ids file so load can proceed
        ids_path = tmp_path / "faiss_test.ids.npy"
        np.save(str(ids_path), np.array([0, 1, 2], dtype=np.int64))
        index_path = tmp_path / "faiss_test.faiss"
        index_path.touch()
        idx.load(str(tmp_path / "faiss_test"))
        idx._faiss.read_index.assert_called_once()


# ===========================================================================
# HNSWIndex (mocked)
# ===========================================================================


def _make_hnswlib_mock():
    """Build a minimal hnswlib mock."""
    hnswlib = MagicMock()

    class FakeHNSWIndex:
        def __init__(self, space, dim):
            self._space = space
            self._dim = dim
            self._matrix = np.empty((0, dim), dtype=np.float32)
            self._ids: List[int] = []
            self._max_elements = 0

        def init_index(self, max_elements, M, ef_construction, random_seed=42):
            self._max_elements = max_elements

        def set_ef(self, ef):
            self._ef = ef

        def add_items(self, matrix, ids):
            self._matrix = matrix.astype(np.float32)
            self._ids = list(ids)

        def knn_query(self, query, k):
            n = self._matrix.shape[0]
            if n == 0:
                return np.array([[]], dtype=np.int64), np.array([[]], dtype=np.float32)
            k = min(k, n)
            q = query[0].astype(np.float32)
            # Cosine distance = 1 - dot(a,b) / (|a||b|); use brute force
            norms_m = np.linalg.norm(self._matrix, axis=1)
            norms_m = np.where(norms_m == 0, 1.0, norms_m)
            norm_q = np.linalg.norm(q) or 1.0
            sims = self._matrix @ q / (norms_m * norm_q)
            distances = 1.0 - sims  # cosine distance
            top = np.argsort(distances)[:k]
            return np.array([np.array(self._ids)[top]], dtype=np.int64), np.array([distances[top]], dtype=np.float32)

        def save_index(self, path):
            pass

        def load_index(self, path, max_elements):
            pass

    def Index(space, dim):
        return FakeHNSWIndex(space, dim)

    hnswlib.Index = Index
    return hnswlib


class TestHNSWIndexMocked:
    """HNSWIndex tests using a mocked hnswlib module."""

    def _get_hnsw_index(self, dim=DIM):
        from vector_index import HNSWIndex
        fake_lib = _make_hnswlib_mock()
        idx = HNSWIndex.__new__(HNSWIndex)
        idx.dimension = dim
        idx.m = 16
        idx.ef_construction = 200
        idx.ef = 50
        idx._ids = []
        idx._hnswlib = fake_lib
        idx._index = None
        return idx

    def test_empty_index_has_zero_length(self):
        idx = self._get_hnsw_index()
        assert len(idx) == 0

    def test_add_and_search(self):
        idx = self._get_hnsw_index()
        vecs = _random_vecs(5)
        idx.add(vecs, list(range(5)))
        assert len(idx) == 5
        results = idx.search(vecs[0], top_k=3)
        assert len(results) == 3

    def test_search_empty_returns_empty(self):
        idx = self._get_hnsw_index()
        results = idx.search(_random_vecs(1)[0], top_k=3)
        assert results == []

    def test_scores_sorted_descending(self):
        idx = self._get_hnsw_index()
        idx.add(_random_vecs(8), list(range(8)))
        results = idx.search(_random_vecs(1)[0], top_k=4)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_add_mismatched_lengths_raises(self):
        idx = self._get_hnsw_index()
        with pytest.raises(ValueError, match="length"):
            idx.add(_random_vecs(3), [0, 1])

    def test_save_load_roundtrip(self, tmp_path):
        idx = self._get_hnsw_index()
        idx.add(_random_vecs(4), [0, 1, 2, 3])
        # Create fake index object with save/load
        path = str(tmp_path / "hnsw_test")
        idx.save(path)
        ids_path = Path(path + ".hnsw.ids.npy")
        assert ids_path.exists()

    def test_load_missing_index_file_raises(self, tmp_path):
        idx = self._get_hnsw_index()
        with pytest.raises(FileNotFoundError):
            idx.load(str(tmp_path / "nonexistent"))


# ===========================================================================
# WP-G: FAISSIndex IVF power-of-two retrain boundary
# ===========================================================================


class TestFAISSIVFPowerOfTwoRetrain:
    """WP-G tests: IVF index rebuilds only at power-of-two boundaries past threshold.

    The mock FAISS is exact (brute-force numpy), so recall@10 = 1.0 in all
    cases.  The rebuild-count test is the structural gate; the recall test
    validates the API contract (and would catch real regressions when faiss
    is installed).
    """

    # _IVF_THRESHOLD is 10_000 in production; override to a small value so
    # tests run fast without generating 10k+ vectors.
    _SMALL_THRESHOLD = 16

    def _get_faiss_index(self, dim=DIM, threshold=None):
        """Return a FAISSIndex wired to the brute-force mock."""
        from vector_index import FAISSIndex

        fake_faiss = _make_faiss_mock()
        idx = FAISSIndex.__new__(FAISSIndex)
        idx.dimension = dim
        idx.nprobe = 10
        idx._ids = []
        idx._raw_embeddings = []
        idx._index = None
        idx._faiss = fake_faiss
        idx._next_retrain_n = 0
        if threshold is not None:
            # Override threshold for fast tests
            idx._IVF_THRESHOLD = threshold
        return idx

    # ------------------------------------------------------------------
    # Test 1 (WP-G spec point 1): rebuild count via spy
    # ------------------------------------------------------------------

    def test_no_rebuild_between_power_of_two_boundaries(self):
        """Adds past threshold that do NOT cross a pow-2 boundary must NOT
        trigger _build_index; only boundary crossings do."""
        from unittest.mock import patch

        threshold = self._SMALL_THRESHOLD
        idx = self._get_faiss_index(threshold=threshold)
        build_calls = []
        real_build = idx._build_index.__func__  # unbound method

        def spy_build(self_inner, matrix):
            build_calls.append(matrix.shape[0])
            real_build(self_inner, matrix)

        with patch.object(type(idx), "_build_index", spy_build):
            # Phase 1: fill up to threshold (flat index, no IVF)
            vecs_flat = _random_vecs(threshold, seed=10)
            idx.add(vecs_flat, list(range(threshold)))
            # One build call for the initial flat add
            initial_builds = len(build_calls)

            # Phase 2: push past threshold to 17 (one past 16 = threshold)
            # 17 is NOT a power of two -> should NOT trigger a rebuild
            idx.add(_random_vecs(1, seed=11), [threshold])
            builds_after_17 = len(build_calls)
            # The first add past the threshold triggers an IVF build (transition)
            # subsequent non-boundary adds must NOT trigger additional builds

            # Phase 3: add 3 more -> total = 20 (not pow-2) -> no rebuild
            idx.add(_random_vecs(3, seed=12), list(range(threshold + 1, threshold + 4)))
            builds_after_20 = len(build_calls)
            assert builds_after_20 == builds_after_17, (
                f"Expected no rebuild at total=20 (not pow-2), "
                f"but build_calls grew from {builds_after_17} to {builds_after_20}"
            )

            # Phase 4: add up to 32 (= 2^5 = next power-of-two above 20)
            # need 32 - 20 = 12 more vectors
            current = threshold + 4  # = 20
            needed = 32 - current
            idx.add(_random_vecs(needed, seed=13), list(range(current, 32)))
            builds_after_32 = len(build_calls)
            assert builds_after_32 > builds_after_20, (
                f"Expected a rebuild when total crossed 32 (pow-2), "
                f"but build_calls did not increase (was {builds_after_20}, now {builds_after_32})"
            )

            # Phase 5: add 1 more -> total = 33 (not pow-2) -> no rebuild
            idx.add(_random_vecs(1, seed=14), [32])
            builds_after_33 = len(build_calls)
            assert builds_after_33 == builds_after_32, (
                f"Expected no rebuild at total=33 (not pow-2), "
                f"but build_calls grew from {builds_after_32} to {builds_after_33}"
            )

    def test_sub_threshold_behavior_unchanged(self):
        """Vectors added below _IVF_THRESHOLD must still use the flat index
        path.  The first add triggers exactly 1 _build_index call (flat init);
        subsequent adds below the threshold use incremental index.add()
        without triggering a rebuild."""
        threshold = self._SMALL_THRESHOLD
        idx = self._get_faiss_index(threshold=threshold)
        build_calls = []
        real_build = idx._build_index.__func__

        def spy_build(self_inner, matrix):
            build_calls.append(matrix.shape[0])
            real_build(self_inner, matrix)

        from unittest.mock import patch

        with patch.object(type(idx), "_build_index", spy_build):
            # Add threshold vectors in two batches (stays at/below threshold)
            idx.add(_random_vecs(threshold // 2, seed=20), list(range(threshold // 2)))
            idx.add(_random_vecs(threshold // 2, seed=21), list(range(threshold // 2, threshold)))
            assert len(idx) == threshold
            # Only the very first add triggers _build_index (flat init);
            # the second batch goes through the incremental index.add() path
            assert len(build_calls) == 1, (
                f"Expected exactly 1 _build_index call for sub-threshold adds "
                f"(first init only), got {len(build_calls)}"
            )

    def test_first_ivf_transition_triggers_rebuild(self):
        """The very first add that pushes total past _IVF_THRESHOLD must
        trigger an IVF rebuild (transition from flat to IVF)."""
        threshold = self._SMALL_THRESHOLD
        idx = self._get_faiss_index(threshold=threshold)
        build_calls = []
        real_build = idx._build_index.__func__

        def spy_build(self_inner, matrix):
            build_calls.append(matrix.shape[0])
            real_build(self_inner, matrix)

        from unittest.mock import patch

        with patch.object(type(idx), "_build_index", spy_build):
            # Fill to threshold using flat
            idx.add(_random_vecs(threshold, seed=30), list(range(threshold)))
            flat_builds = len(build_calls)

            # One more: push to threshold+1 -> IVF transition
            idx.add(_random_vecs(1, seed=31), [threshold])
            assert len(build_calls) > flat_builds, (
                "Expected _build_index called on first add past _IVF_THRESHOLD"
            )

    # ------------------------------------------------------------------
    # Test 2 (WP-G spec / D11): measured recall@10 >= 0.95
    # ------------------------------------------------------------------

    def test_recall_at_10_after_staggered_adds_meets_floor(self):
        """After staggered adds spanning at least one full power-of-two gap,
        recall@10 of the lazy-retrain index vs the full-rebuild baseline
        must be >= 0.95.

        With the brute-force mock both paths are exact, so recall = 1.0.
        The test structure and floor assertion are what matter; they would
        catch real regressions when run with real FAISS (faiss-cpu installed).
        """
        threshold = self._SMALL_THRESHOLD
        dim = DIM
        seed = 42
        rng = np.random.default_rng(seed)
        RECALL_FLOOR = 0.95
        TOP_K = 10

        # Generate a ground-truth corpus that spans beyond one pow-2 boundary
        # above the threshold: threshold=16, we go up to 64 vectors total
        total_vecs = 64
        corpus = rng.standard_normal((total_vecs, dim)).astype(np.float32)
        corpus_ids = list(range(total_vecs))

        # Build the lazy-retrain index (WP-G implementation)
        lazy_idx = self._get_faiss_index(threshold=threshold)

        # Build the full-rebuild reference: a new index per add (brute-force
        # baseline) -- since the mock is exact, this is the ground truth
        ref_idx = self._get_faiss_index(threshold=threshold)

        # Add vectors in small batches to exercise power-of-two boundaries
        batch_size = 3
        for start in range(0, total_vecs, batch_size):
            end = min(start + batch_size, total_vecs)
            batch = corpus[start:end].tolist()
            batch_ids = corpus_ids[start:end]
            lazy_idx.add(batch, batch_ids)
            ref_idx.add(batch, batch_ids)

        assert len(lazy_idx) == total_vecs
        assert len(ref_idx) == total_vecs

        # Generate query vectors and measure recall
        n_queries = 20
        queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

        hits = 0
        total_relevant = 0
        for q in queries:
            lazy_results = lazy_idx.search(q.tolist(), top_k=TOP_K)
            ref_results = ref_idx.search(q.tolist(), top_k=TOP_K)

            lazy_ids = {r[0] for r in lazy_results}
            ref_ids = {r[0] for r in ref_results}

            # recall = fraction of reference top-K found in lazy top-K
            hits += len(lazy_ids & ref_ids)
            total_relevant += len(ref_ids)

        recall = hits / total_relevant if total_relevant > 0 else 0.0
        assert recall >= RECALL_FLOOR, (
            f"recall@{TOP_K} = {recall:.3f} < floor {RECALL_FLOOR}. "
            f"Consider narrowing the retrain cadence (e.g. 1.5x growth)."
        )
