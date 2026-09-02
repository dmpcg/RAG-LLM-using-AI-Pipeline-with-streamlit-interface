"""Tests for ml/registry.py – ModelMetadata, ModelRegistry, TrainingPipeline."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from ml.registry import ModelMetadata, ModelRegistry, TrainingPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_registry(tmp_path):
    """Return a ModelRegistry backed by a temporary directory."""
    return ModelRegistry(registry_dir=str(tmp_path))


@pytest.fixture
def sample_metadata() -> ModelMetadata:
    return ModelMetadata(
        model_id=str(uuid.uuid4()),
        model_name="TestClassifier",
        model_type="classifier",
        version="1.0.0",
        created_at=datetime.now(timezone.utc).isoformat(),
        metrics={"accuracy": 0.9, "f1": 0.88},
        feature_names=["feat_a", "feat_b"],
        parameters={"model_type": "random_forest"},
        description="Test model",
        status="active",
    )


@pytest.fixture
def dummy_model() -> dict:
    """A trivial serialisable object that stands in for a real ML model."""
    return {"type": "dummy", "coef": [1.0, 2.0]}


@pytest.fixture
def pipeline(tmp_registry) -> TrainingPipeline:
    return TrainingPipeline(registry=tmp_registry)


# ---------------------------------------------------------------------------
# ModelMetadata
# ---------------------------------------------------------------------------


class TestModelMetadata:
    def test_creation_and_fields(self):
        meta = ModelMetadata(
            model_id="abc-123",
            model_name="MyModel",
            model_type="classifier",
            version="2.1.0",
            created_at="2024-01-01T00:00:00+00:00",
            metrics={"accuracy": 0.95},
            feature_names=["x1", "x2"],
            parameters={"n_estimators": 100},
            description="A test model",
            status="staging",
        )
        assert meta.model_id == "abc-123"
        assert meta.model_name == "MyModel"
        assert meta.model_type == "classifier"
        assert meta.version == "2.1.0"
        assert meta.metrics == {"accuracy": 0.95}
        assert meta.feature_names == ["x1", "x2"]
        assert meta.parameters == {"n_estimators": 100}
        assert meta.description == "A test model"
        assert meta.status == "staging"

    def test_defaults(self):
        meta = ModelMetadata(
            model_id="id1",
            model_name="M",
            model_type="forecaster",
            version="1.0",
            created_at="2024-01-01T00:00:00+00:00",
        )
        assert meta.metrics == {}
        assert meta.feature_names == []
        assert meta.parameters == {}
        assert meta.description == ""
        assert meta.status == "active"

    def test_to_dict_and_from_dict(self, sample_metadata):
        d = sample_metadata.to_dict()
        assert isinstance(d, dict)
        restored = ModelMetadata.from_dict(d)
        assert restored.model_id == sample_metadata.model_id
        assert restored.metrics == sample_metadata.metrics
        assert restored.feature_names == sample_metadata.feature_names


# ---------------------------------------------------------------------------
# ModelRegistry – register / load
# ---------------------------------------------------------------------------


class TestModelRegistryRegisterLoad:
    def test_register_returns_model_id(self, tmp_registry, dummy_model, sample_metadata):
        model_id = tmp_registry.register(dummy_model, sample_metadata)
        assert model_id == sample_metadata.model_id

    def test_register_creates_model_file(self, tmp_path, dummy_model, sample_metadata):
        registry = ModelRegistry(registry_dir=str(tmp_path))
        model_id = registry.register(dummy_model, sample_metadata)
        artifact = tmp_path / model_id / "model.joblib"
        assert artifact.exists()

    def test_register_creates_metadata_file(self, tmp_path, dummy_model, sample_metadata):
        registry = ModelRegistry(registry_dir=str(tmp_path))
        model_id = registry.register(dummy_model, sample_metadata)
        meta_file = tmp_path / model_id / "metadata.json"
        assert meta_file.exists()

    def test_metadata_json_is_valid(self, tmp_path, dummy_model, sample_metadata):
        registry = ModelRegistry(registry_dir=str(tmp_path))
        model_id = registry.register(dummy_model, sample_metadata)
        meta_file = tmp_path / model_id / "metadata.json"
        data = json.loads(meta_file.read_text())
        assert data["model_name"] == sample_metadata.model_name
        assert data["status"] == "active"

    def test_load_returns_model_and_metadata(self, tmp_registry, dummy_model, sample_metadata):
        model_id = tmp_registry.register(dummy_model, sample_metadata)
        loaded_model, loaded_meta = tmp_registry.load(model_id)
        assert loaded_model == dummy_model
        assert loaded_meta.model_id == model_id

    def test_load_metadata_fields_intact(self, tmp_registry, dummy_model, sample_metadata):
        model_id = tmp_registry.register(dummy_model, sample_metadata)
        _, meta = tmp_registry.load(model_id)
        assert meta.model_type == sample_metadata.model_type
        assert meta.metrics == sample_metadata.metrics
        assert meta.feature_names == sample_metadata.feature_names

    def test_load_nonexistent_raises(self, tmp_registry):
        with pytest.raises(FileNotFoundError):
            tmp_registry.load("nonexistent-id-xyz")

    def test_register_assigns_uuid_when_id_blank(self, tmp_path, dummy_model):
        registry = ModelRegistry(registry_dir=str(tmp_path))
        meta = ModelMetadata(
            model_id="",
            model_name="NoBlanks",
            model_type="classifier",
            version="1.0",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        model_id = registry.register(dummy_model, meta)
        assert model_id  # non-empty
        assert meta.model_id == model_id


# ---------------------------------------------------------------------------
# ModelRegistry – list_models
# ---------------------------------------------------------------------------


class TestModelRegistryListModels:
    def _make_meta(self, model_type: str, status: str, offset_seconds: int = 0) -> ModelMetadata:
        ts = (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()
        return ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name="M",
            model_type=model_type,
            version="1.0",
            created_at=ts,
            status=status,
        )

    def test_list_all_models(self, tmp_registry, dummy_model):
        tmp_registry.register(dummy_model, self._make_meta("classifier", "active", 0))
        tmp_registry.register(dummy_model, self._make_meta("forecaster", "active", 1))
        tmp_registry.register(dummy_model, self._make_meta("clusterer", "archived", 2))
        models = tmp_registry.list_models()
        assert len(models) == 3

    def test_list_models_filter_by_type(self, tmp_registry, dummy_model):
        tmp_registry.register(dummy_model, self._make_meta("classifier", "active"))
        tmp_registry.register(dummy_model, self._make_meta("forecaster", "active"))
        classifiers = tmp_registry.list_models(model_type="classifier")
        assert all(m.model_type == "classifier" for m in classifiers)
        assert len(classifiers) == 1

    def test_list_models_filter_by_status(self, tmp_registry, dummy_model):
        tmp_registry.register(dummy_model, self._make_meta("classifier", "active"))
        tmp_registry.register(dummy_model, self._make_meta("classifier", "archived"))
        active = tmp_registry.list_models(status="active")
        assert all(m.status == "active" for m in active)

    def test_list_models_filter_by_type_and_status(self, tmp_registry, dummy_model):
        tmp_registry.register(dummy_model, self._make_meta("classifier", "active"))
        tmp_registry.register(dummy_model, self._make_meta("classifier", "archived"))
        tmp_registry.register(dummy_model, self._make_meta("forecaster", "active"))
        result = tmp_registry.list_models(model_type="classifier", status="active")
        assert len(result) == 1
        assert result[0].model_type == "classifier"
        assert result[0].status == "active"

    def test_list_models_sorted_newest_first(self, tmp_registry, dummy_model):
        tmp_registry.register(dummy_model, self._make_meta("classifier", "active", offset_seconds=0))
        tmp_registry.register(dummy_model, self._make_meta("classifier", "active", offset_seconds=10))
        tmp_registry.register(dummy_model, self._make_meta("classifier", "active", offset_seconds=5))
        models = tmp_registry.list_models()
        timestamps = [m.created_at for m in models]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_models_empty_registry(self, tmp_registry):
        assert tmp_registry.list_models() == []


# ---------------------------------------------------------------------------
# ModelRegistry – get_active
# ---------------------------------------------------------------------------


class TestModelRegistryGetActive:
    def test_get_active_returns_latest(self, tmp_path, dummy_model):
        registry = ModelRegistry(registry_dir=str(tmp_path))
        older = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name="Old",
            model_type="classifier",
            version="1.0",
            created_at=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            status="active",
        )
        newer = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name="New",
            model_type="classifier",
            version="2.0",
            created_at=datetime.now(timezone.utc).isoformat(),
            status="active",
        )
        registry.register({"version": 1}, older)
        registry.register({"version": 2}, newer)

        _, meta = registry.get_active("classifier")
        assert meta.model_name == "New"

    def test_get_active_raises_when_no_active(self, tmp_registry, dummy_model):
        meta = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name="Archived",
            model_type="classifier",
            version="1.0",
            created_at=datetime.now(timezone.utc).isoformat(),
            status="archived",
        )
        tmp_registry.register(dummy_model, meta)
        with pytest.raises(LookupError):
            tmp_registry.get_active("classifier")

    def test_get_active_raises_when_type_missing(self, tmp_registry):
        with pytest.raises(LookupError):
            tmp_registry.get_active("nonexistent_type")


# ---------------------------------------------------------------------------
# ModelRegistry – update_status / delete / compare
# ---------------------------------------------------------------------------


class TestModelRegistryOperations:
    def test_update_status_persists(self, tmp_registry, dummy_model, sample_metadata):
        model_id = tmp_registry.register(dummy_model, sample_metadata)
        tmp_registry.update_status(model_id, "archived")
        _, meta = tmp_registry.load(model_id)
        assert meta.status == "archived"

    def test_update_status_to_staging(self, tmp_registry, dummy_model, sample_metadata):
        model_id = tmp_registry.register(dummy_model, sample_metadata)
        tmp_registry.update_status(model_id, "staging")
        _, meta = tmp_registry.load(model_id)
        assert meta.status == "staging"

    def test_update_status_nonexistent_raises(self, tmp_registry):
        with pytest.raises(FileNotFoundError):
            tmp_registry.update_status("ghost-id", "archived")

    def test_delete_removes_directory(self, tmp_path, dummy_model, sample_metadata):
        registry = ModelRegistry(registry_dir=str(tmp_path))
        model_id = registry.register(dummy_model, sample_metadata)
        model_dir = tmp_path / model_id
        assert model_dir.exists()
        registry.delete(model_id)
        assert not model_dir.exists()

    def test_delete_nonexistent_raises(self, tmp_registry):
        with pytest.raises(FileNotFoundError):
            tmp_registry.delete("ghost-id")

    def test_delete_then_list_is_empty(self, tmp_registry, dummy_model, sample_metadata):
        model_id = tmp_registry.register(dummy_model, sample_metadata)
        tmp_registry.delete(model_id)
        assert tmp_registry.list_models() == []

    def test_compare_returns_diff_for_each_metric(self, tmp_path, dummy_model):
        registry = ModelRegistry(registry_dir=str(tmp_path))
        ts = datetime.now(timezone.utc).isoformat()

        meta_a = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name="A",
            model_type="classifier",
            version="1.0",
            created_at=ts,
            metrics={"accuracy": 0.9, "f1": 0.88, "mae": 0.1},
        )
        meta_b = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name="B",
            model_type="classifier",
            version="2.0",
            created_at=ts,
            metrics={"accuracy": 0.85, "f1": 0.92, "mae": 0.05},
        )
        id_a = registry.register(dummy_model, meta_a)
        id_b = registry.register(dummy_model, meta_b)

        comparison = registry.compare(id_a, id_b)

        assert "accuracy" in comparison
        assert "f1" in comparison
        assert "mae" in comparison

        # accuracy: A=0.9 > B=0.85, higher is better => A wins
        assert comparison["accuracy"]["better"] == "a"
        assert comparison["accuracy"]["a"] == pytest.approx(0.9)
        assert comparison["accuracy"]["b"] == pytest.approx(0.85)

        # f1: B=0.92 > A=0.88, higher is better => B wins
        assert comparison["f1"]["better"] == "b"

        # mae: lower is better; B=0.05 < A=0.1 => B wins
        assert comparison["mae"]["better"] == "b"

    def test_compare_diff_value(self, tmp_path, dummy_model):
        registry = ModelRegistry(registry_dir=str(tmp_path))
        ts = datetime.now(timezone.utc).isoformat()
        meta_a = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name="A",
            model_type="c",
            version="1",
            created_at=ts,
            metrics={"accuracy": 0.8},
        )
        meta_b = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name="B",
            model_type="c",
            version="2",
            created_at=ts,
            metrics={"accuracy": 0.6},
        )
        id_a = registry.register(dummy_model, meta_a)
        id_b = registry.register(dummy_model, meta_b)
        result = registry.compare(id_a, id_b)
        assert result["accuracy"]["diff"] == pytest.approx(0.2)

    def test_compare_tie(self, tmp_path, dummy_model):
        registry = ModelRegistry(registry_dir=str(tmp_path))
        ts = datetime.now(timezone.utc).isoformat()
        meta_a = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name="A",
            model_type="c",
            version="1",
            created_at=ts,
            metrics={"accuracy": 0.9},
        )
        meta_b = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name="B",
            model_type="c",
            version="2",
            created_at=ts,
            metrics={"accuracy": 0.9},
        )
        id_a = registry.register(dummy_model, meta_a)
        id_b = registry.register(dummy_model, meta_b)
        result = registry.compare(id_a, id_b)
        assert result["accuracy"]["better"] == "tie"

    def test_compare_only_common_metrics(self, tmp_path, dummy_model):
        registry = ModelRegistry(registry_dir=str(tmp_path))
        ts = datetime.now(timezone.utc).isoformat()
        meta_a = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name="A",
            model_type="c",
            version="1",
            created_at=ts,
            metrics={"accuracy": 0.9, "only_a": 1.0},
        )
        meta_b = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name="B",
            model_type="c",
            version="2",
            created_at=ts,
            metrics={"accuracy": 0.85, "only_b": 2.0},
        )
        id_a = registry.register(dummy_model, meta_a)
        id_b = registry.register(dummy_model, meta_b)
        result = registry.compare(id_a, id_b)
        assert "accuracy" in result
        assert "only_a" not in result
        assert "only_b" not in result


# ---------------------------------------------------------------------------
# Multiple models coexistence
# ---------------------------------------------------------------------------


class TestRegistryMultipleModels:
    def test_multiple_models_in_same_registry(self, tmp_registry, dummy_model):
        ids = []
        for i in range(5):
            meta = ModelMetadata(
                model_id=str(uuid.uuid4()),
                model_name=f"Model{i}",
                model_type="classifier",
                version=f"1.{i}",
                created_at=(datetime.now(timezone.utc) + timedelta(seconds=i)).isoformat(),
            )
            ids.append(tmp_registry.register(dummy_model, meta))

        assert len(tmp_registry.list_models()) == 5

    def test_mixed_types_listing(self, tmp_registry, dummy_model):
        ts = datetime.now(timezone.utc).isoformat()
        for mtype in ("classifier", "forecaster", "clusterer"):
            meta = ModelMetadata(
                model_id=str(uuid.uuid4()),
                model_name="M",
                model_type=mtype,
                version="1.0",
                created_at=ts,
            )
            tmp_registry.register(dummy_model, meta)

        assert len(tmp_registry.list_models(model_type="forecaster")) == 1
        assert len(tmp_registry.list_models(model_type="classifier")) == 1
        assert len(tmp_registry.list_models(model_type="clusterer")) == 1


# ---------------------------------------------------------------------------
# TrainingPipeline – classifier
# ---------------------------------------------------------------------------


class TestTrainingPipelineClassifier:
    def _make_data(self, n: int = 30):
        """Return synthetic features and labels for classifier training."""
        import random

        random.seed(42)
        features = [[random.gauss(0, 1) for _ in range(5)] for _ in range(n)]
        labels = [0 if features[i][0] > 0 else 1 for i in range(n)]
        return features, labels

    def test_train_classifier_returns_model_id(self, pipeline):
        features, labels = self._make_data(40)
        feature_names = ["f1", "f2", "f3", "f4", "f5"]
        model_id = pipeline.train_classifier(features, labels, feature_names)
        assert model_id  # non-empty string

    def test_train_classifier_model_loadable(self, pipeline):
        features, labels = self._make_data(40)
        feature_names = ["f1", "f2", "f3", "f4", "f5"]
        model_id = pipeline.train_classifier(features, labels, feature_names)
        model, meta = pipeline.registry.load(model_id)
        assert hasattr(model, "predict")

    def test_train_classifier_metadata_correct(self, pipeline):
        features, labels = self._make_data(40)
        feature_names = ["f1", "f2", "f3", "f4", "f5"]
        model_id = pipeline.train_classifier(
            features,
            labels,
            feature_names,
            model_type="logistic_regression",
            model_name="LRModel",
            version="2.0",
        )
        _, meta = pipeline.registry.load(model_id)
        assert meta.model_type == "classifier"
        assert meta.model_name == "LRModel"
        assert meta.version == "2.0"
        assert meta.parameters["model_type"] == "logistic_regression"
        assert meta.feature_names == feature_names

    def test_train_classifier_metrics_recorded(self, pipeline):
        features, labels = self._make_data(40)
        feature_names = ["f1", "f2", "f3", "f4", "f5"]
        model_id = pipeline.train_classifier(features, labels, feature_names)
        _, meta = pipeline.registry.load(model_id)
        assert "accuracy" in meta.metrics
        assert 0.0 <= meta.metrics["accuracy"] <= 1.0

    def test_train_classifier_status_active(self, pipeline):
        features, labels = self._make_data(40)
        feature_names = ["f1", "f2", "f3", "f4", "f5"]
        model_id = pipeline.train_classifier(features, labels, feature_names)
        _, meta = pipeline.registry.load(model_id)
        assert meta.status == "active"


# ---------------------------------------------------------------------------
# TrainingPipeline – forecaster
# ---------------------------------------------------------------------------


class TestTrainingPipelineForecaster:
    def _make_series(self, n: int = 20) -> list[float]:
        import math

        return [math.sin(i * 0.5) + i * 0.05 for i in range(n)]

    def test_train_forecaster_ensemble_returns_model_id(self, pipeline):
        values = self._make_series(20)
        model_id = pipeline.train_forecaster(values, method="ensemble")
        assert model_id

    def test_train_forecaster_ar_returns_model_id(self, pipeline):
        values = self._make_series(20)
        model_id = pipeline.train_forecaster(values, method="ar")
        assert model_id

    def test_train_forecaster_exponential_returns_model_id(self, pipeline):
        values = self._make_series(20)
        model_id = pipeline.train_forecaster(values, method="exponential")
        assert model_id

    def test_train_forecaster_model_loadable(self, pipeline):
        values = self._make_series(20)
        model_id = pipeline.train_forecaster(values, method="ensemble")
        model, meta = pipeline.registry.load(model_id)
        assert hasattr(model, "predict")

    def test_train_forecaster_metadata_correct(self, pipeline):
        values = self._make_series(20)
        model_id = pipeline.train_forecaster(
            values,
            method="ar",
            model_name="ARModel",
            version="3.0",
        )
        _, meta = pipeline.registry.load(model_id)
        assert meta.model_type == "forecaster"
        assert meta.model_name == "ARModel"
        assert meta.version == "3.0"
        assert meta.parameters["method"] == "ar"

    def test_train_forecaster_invalid_method_raises(self, pipeline):
        values = self._make_series(20)
        with pytest.raises(ValueError, match="Unknown method"):
            pipeline.train_forecaster(values, method="invalid_method")


# ---------------------------------------------------------------------------
# TrainingPipeline – retrain_if_stale
# ---------------------------------------------------------------------------


class TestRetrainIfStale:
    def _register_with_age(self, registry: ModelRegistry, age_days: int) -> str:
        """Register a dummy model whose created_at is *age_days* ago."""
        ts = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        meta = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name="AgedModel",
            model_type="classifier",
            version="1.0",
            created_at=ts,
        )
        return registry.register({"dummy": True}, meta)

    def test_stale_model_returns_model_id(self, pipeline):
        old_id = self._register_with_age(pipeline.registry, age_days=60)
        result = pipeline.retrain_if_stale(old_id, max_age_days=30)
        assert result == old_id

    def test_fresh_model_returns_none(self, pipeline):
        fresh_id = self._register_with_age(pipeline.registry, age_days=5)
        result = pipeline.retrain_if_stale(fresh_id, max_age_days=30)
        assert result is None

    def test_exactly_at_boundary_is_stale(self, pipeline):
        """A model that is exactly max_age_days old is considered stale."""
        boundary_id = self._register_with_age(pipeline.registry, age_days=30)
        result = pipeline.retrain_if_stale(boundary_id, max_age_days=30)
        assert result == boundary_id

    def test_one_day_under_boundary_is_fresh(self, pipeline):
        fresh_id = self._register_with_age(pipeline.registry, age_days=29)
        result = pipeline.retrain_if_stale(fresh_id, max_age_days=30)
        assert result is None
