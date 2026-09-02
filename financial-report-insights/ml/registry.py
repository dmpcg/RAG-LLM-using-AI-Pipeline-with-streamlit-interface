"""Model registry for ML lifecycle management.

Provides ModelMetadata, ModelRegistry, and TrainingPipeline to handle
train -> evaluate -> register -> load -> compare workflows.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib

# Regex to reject path traversal and unsafe characters in model_id
_SAFE_MODEL_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")


# ---------------------------------------------------------------------------
# ModelMetadata
# ---------------------------------------------------------------------------


@dataclass
class ModelMetadata:
    """Metadata associated with a registered model artifact."""

    model_id: str
    model_name: str
    model_type: str  # "classifier", "forecaster", "clusterer"
    version: str
    created_at: str  # ISO-8601 timestamp
    metrics: dict = field(default_factory=dict)
    feature_names: list = field(default_factory=list)
    parameters: dict = field(default_factory=dict)
    description: str = ""
    status: str = "active"  # "active" | "staging" | "archived"

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        return cls(**data)


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Persist and retrieve ML model artefacts together with their metadata.

    Parameters
    ----------
    registry_dir:
        Root directory where models are stored.  Each model is saved in its
        own sub-directory named by ``model_id``.
    """

    def __init__(self, registry_dir: str = "./data/models") -> None:
        self._root = Path(registry_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_model_id(model_id: str) -> None:
        """Reject model IDs that could escape the registry directory."""
        if not model_id or not _SAFE_MODEL_ID.match(model_id):
            raise ValueError(f"Invalid model_id {model_id!r}. Must match [a-zA-Z0-9][a-zA-Z0-9._-]*")
        if ".." in model_id:
            raise ValueError(f"Path traversal detected in model_id: {model_id!r}")

    def _model_dir(self, model_id: str) -> Path:
        self._validate_model_id(model_id)
        resolved = (self._root / model_id).resolve()
        if not str(resolved).startswith(str(self._root.resolve())):
            raise ValueError(f"model_id escapes registry root: {model_id!r}")
        return self._root / model_id

    def _metadata_path(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "metadata.json"

    def _artifact_path(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "model.joblib"

    def _write_metadata(self, metadata: ModelMetadata) -> None:
        path = self._metadata_path(metadata.model_id)
        path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")

    def _read_metadata(self, model_id: str) -> ModelMetadata:
        path = self._metadata_path(model_id)
        if not path.exists():
            raise FileNotFoundError(f"No model registered with id '{model_id}'")
        data = json.loads(path.read_text(encoding="utf-8"))
        return ModelMetadata.from_dict(data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, model: Any, metadata: ModelMetadata) -> str:
        """Save *model* and *metadata* to disk and return ``model_id``.

        Parameters
        ----------
        model:
            Any serialisable Python object (sklearn estimator, custom class …).
        metadata:
            A fully-populated :class:`ModelMetadata` instance.  The
            ``model_id`` field is used as the directory name; if it is blank a
            new UUID is assigned.

        Returns
        -------
        str
            The ``model_id`` that was persisted.
        """
        if not metadata.model_id:
            metadata.model_id = str(uuid.uuid4())

        model_dir = self._model_dir(metadata.model_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = self._artifact_path(metadata.model_id)
        joblib.dump(model, artifact_path)

        # Store SHA-256 hash for integrity verification on load
        sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        metadata.parameters["_artifact_sha256"] = sha256

        self._write_metadata(metadata)

        return metadata.model_id

    def load(self, model_id: str) -> tuple[Any, ModelMetadata]:
        """Load a model artefact together with its metadata.

        Parameters
        ----------
        model_id:
            Identifier returned by :meth:`register`.

        Returns
        -------
        tuple[Any, ModelMetadata]
            ``(model_object, metadata)``

        Raises
        ------
        FileNotFoundError
            When no model with *model_id* exists in the registry.
        """
        artifact = self._artifact_path(model_id)
        if not artifact.exists():
            raise FileNotFoundError(f"No model artefact found for id '{model_id}'")

        # Verify artifact integrity before deserializing
        metadata = self._read_metadata(model_id)
        expected_hash = metadata.parameters.get("_artifact_sha256")
        if expected_hash:
            actual_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Artifact integrity check failed for model '{model_id}': "
                    f"expected SHA-256 {expected_hash[:16]}..., got {actual_hash[:16]}..."
                )

        model = joblib.load(artifact)
        return model, metadata

    def list_models(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[ModelMetadata]:
        """Return all registered models, optionally filtered.

        Results are sorted by ``created_at`` descending (newest first).

        Parameters
        ----------
        model_type:
            When given, only models with this ``model_type`` are returned.
        status:
            When given, only models with this ``status`` are returned.
        """
        results: list[ModelMetadata] = []

        for subdir in self._root.iterdir():
            if not subdir.is_dir():
                continue
            meta_file = subdir / "metadata.json"
            if not meta_file.exists():
                continue
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                meta = ModelMetadata.from_dict(data)
            except (json.JSONDecodeError, TypeError):
                continue

            if model_type is not None and meta.model_type != model_type:
                continue
            if status is not None and meta.status != status:
                continue

            results.append(meta)

        results.sort(key=lambda m: m.created_at, reverse=True)
        return results

    def get_active(self, model_type: str) -> tuple[Any, ModelMetadata]:
        """Return the newest ``active`` model of the given type.

        Parameters
        ----------
        model_type:
            The ``model_type`` field to filter on (e.g. ``"classifier"``).

        Returns
        -------
        tuple[Any, ModelMetadata]

        Raises
        ------
        LookupError
            When no active model of *model_type* exists.
        """
        candidates = self.list_models(model_type=model_type, status="active")
        if not candidates:
            raise LookupError(f"No active model of type '{model_type}' found in registry")
        # list_models already sorted newest-first
        return self.load(candidates[0].model_id)

    def update_status(self, model_id: str, status: str) -> None:
        """Overwrite the ``status`` field for an existing model.

        Parameters
        ----------
        model_id:
            Target model identifier.
        status:
            New status string (e.g. ``"archived"``, ``"staging"``, ``"active"``).

        Raises
        ------
        FileNotFoundError
            When *model_id* is not registered.
        """
        metadata = self._read_metadata(model_id)
        metadata.status = status
        self._write_metadata(metadata)

    def delete(self, model_id: str) -> None:
        """Remove a model and all its files from the registry.

        Parameters
        ----------
        model_id:
            Identifier of the model to delete.

        Raises
        ------
        FileNotFoundError
            When *model_id* is not registered.
        """
        model_dir = self._model_dir(model_id)
        if not model_dir.exists():
            raise FileNotFoundError(f"No model registered with id '{model_id}'")
        shutil.rmtree(model_dir)

    def compare(self, model_id_a: str, model_id_b: str) -> dict[str, dict]:
        """Compare training metrics between two registered models.

        Parameters
        ----------
        model_id_a:
            First model identifier.
        model_id_b:
            Second model identifier.

        Returns
        -------
        dict
            Mapping of metric name to ``{"a": val_a, "b": val_b, "diff": diff,
            "better": "a"|"b"|"tie"}``.  Only metrics present in *both* models
            are included.  For metrics where higher is better the "better" key
            reflects which model scored higher; where lower is better (mae,
            rmse, mape, loss) the smaller value wins.
        """
        _, meta_a = self.load(model_id_a)
        _, meta_b = self.load(model_id_b)

        lower_is_better = {"mae", "rmse", "mape", "loss", "error"}

        common_keys = set(meta_a.metrics) & set(meta_b.metrics)
        result: dict[str, dict] = {}

        for key in sorted(common_keys):
            val_a = float(meta_a.metrics[key])
            val_b = float(meta_b.metrics[key])
            diff = val_a - val_b  # positive means A > B

            if key in lower_is_better:
                if val_a < val_b:
                    better = "a"
                elif val_b < val_a:
                    better = "b"
                else:
                    better = "tie"
            else:
                if val_a > val_b:
                    better = "a"
                elif val_b > val_a:
                    better = "b"
                else:
                    better = "tie"

            result[key] = {
                "a": val_a,
                "b": val_b,
                "diff": diff,
                "better": better,
            }

        return result


# ---------------------------------------------------------------------------
# TrainingPipeline
# ---------------------------------------------------------------------------


class TrainingPipeline:
    """Orchestrate train -> evaluate -> register for common model types.

    Parameters
    ----------
    registry:
        A :class:`ModelRegistry` instance where trained models will be stored.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Classifier training
    # ------------------------------------------------------------------

    def train_classifier(
        self,
        features: list[list[float]],
        labels: list[int],
        feature_names: list[str],
        model_type: str = "random_forest",
        model_name: str = "FinancialDistressClassifier",
        version: str = "1.0.0",
        description: str = "",
        **kwargs: Any,
    ) -> str:
        """Train a :class:`~ml.classifiers.FinancialDistressClassifier`.

        Runs cross-validated training, collects metrics, and registers the
        fitted model in the registry.

        Parameters
        ----------
        features:
            2-D list of shape ``(n_samples, n_features)``.
        labels:
            Binary target list of length ``n_samples``.
        feature_names:
            Names matching the columns of *features*.
        model_type:
            Estimator to use: ``"logistic_regression"``, ``"random_forest"``,
            or ``"gradient_boosting"``.
        model_name:
            Human-readable name stored in metadata.
        version:
            Semantic version string.
        description:
            Optional description stored in metadata.
        **kwargs:
            Reserved for future use.

        Returns
        -------
        str
            ``model_id`` of the newly registered model.
        """
        # Import here to avoid circular imports at module level
        from ml.classifiers import FinancialDistressClassifier

        clf = FinancialDistressClassifier(model_type=model_type)
        metrics = clf.train(features, labels)

        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name=model_name,
            model_type="classifier",
            version=version,
            created_at=self._now_iso(),
            metrics=metrics,
            feature_names=list(feature_names),
            parameters={"model_type": model_type},
            description=description,
            status="active",
        )

        return self.registry.register(clf, metadata)

    # ------------------------------------------------------------------
    # Forecaster training
    # ------------------------------------------------------------------

    def train_forecaster(
        self,
        values: list[float],
        method: str = "ensemble",
        model_name: str = "TimeSeriesForecaster",
        version: str = "1.0.0",
        description: str = "",
        **kwargs: Any,
    ) -> str:
        """Train a time-series forecasting model and register it.

        Parameters
        ----------
        values:
            Historical time series.
        method:
            ``"ensemble"``, ``"ar"``, or ``"exponential"``.
        model_name:
            Human-readable name stored in metadata.
        version:
            Semantic version string.
        description:
            Optional description.
        **kwargs:
            Forwarded to the underlying model's ``fit()`` call.

        Returns
        -------
        str
            ``model_id`` of the newly registered model.
        """
        from ml.forecasting import (
            EnsembleForecaster,
            ExponentialSmoother,
            SimpleARModel,
            walk_forward_validate,
        )

        if method == "ensemble":
            model = EnsembleForecaster()
            model.fit(values)
            metrics: dict[str, float] = dict(model._metrics)
        elif method == "ar":
            order = kwargs.get("order", 2)
            model = SimpleARModel(order=order)
            model.fit(values)
            metrics = walk_forward_validate(
                SimpleARModel(order=order),
                values,
                test_size=min(3, len(values) - order - 1),
            )
        elif method == "exponential":
            es_method = kwargs.get("es_method", "double")
            model = ExponentialSmoother(method=es_method)
            model.fit(values)
            metrics = walk_forward_validate(
                ExponentialSmoother(method=es_method),
                values,
                test_size=min(3, len(values) - 2),
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'ensemble', 'ar', or 'exponential'.")

        # Sanitise metrics: replace nan/inf with None for JSON-safe storage
        clean_metrics: dict[str, Any] = {}
        import math as _math

        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and not _math.isfinite(v)):
                clean_metrics[k] = None
            else:
                clean_metrics[k] = v

        metadata = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name=model_name,
            model_type="forecaster",
            version=version,
            created_at=self._now_iso(),
            metrics=clean_metrics,
            feature_names=[],
            parameters={"method": method},
            description=description,
            status="active",
        )

        return self.registry.register(model, metadata)

    # ------------------------------------------------------------------
    # Staleness check
    # ------------------------------------------------------------------

    def retrain_if_stale(
        self,
        model_id: str,
        max_age_days: int = 30,
    ) -> Optional[str]:
        """Check whether a registered model is older than *max_age_days*.

        Parameters
        ----------
        model_id:
            Identifier of the model to check.
        max_age_days:
            Maximum allowed age in days.

        Returns
        -------
        str or None
            Returns *model_id* if the model is stale and should be retrained,
            otherwise ``None``.
        """
        metadata = self.registry._read_metadata(model_id)

        try:
            created = datetime.fromisoformat(metadata.created_at)
        except ValueError:
            # Cannot parse timestamp; treat as stale
            return model_id

        # Ensure timezone-aware comparison
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_days = (now - created).days

        if age_days >= max_age_days:
            return model_id
        return None
