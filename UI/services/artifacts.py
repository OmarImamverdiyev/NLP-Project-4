from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


IGNORED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
}


@dataclass(slots=True)
class RunRecord:
    task: str
    label: str
    output_dir: Path
    manifest_path: Path | None
    metrics_path: Path | None
    checkpoint_path: Path | None
    metadata: dict[str, Any]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ui_root() -> Path:
    return project_root() / "UI"


def ui_artifact_root() -> Path:
    return ui_root() / "artifacts"


def presets_path() -> Path:
    return ui_root() / "configs" / "defaults.json"


def load_json(path: str | Path, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_name(f".{file_path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(sanitize_for_json(payload), handle, indent=2, ensure_ascii=False)
        temp_path.replace(file_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return file_path


def write_run_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "run_manifest.json"
    return write_json(manifest_path, manifest)


def is_ignored_path(path: Path) -> bool:
    return any(part in IGNORED_PARTS for part in path.parts)


def try_relative_path(path: Path, base: Path | None = None) -> str:
    target_base = base or project_root()
    try:
        return str(path.resolve().relative_to(target_base.resolve()))
    except ValueError:
        return str(path.resolve())


def load_presets() -> dict[str, Any]:
    payload = load_json(presets_path(), default={})
    return payload if isinstance(payload, dict) else {}


def _infer_task_from_metrics(metrics_payload: dict[str, Any], metrics_path: Path) -> str:
    if {"model_name", "dataset_summary", "metrics"}.issubset(metrics_payload.keys()):
        return "sentiment"
    if {"history", "data_summary"}.issubset(metrics_payload.keys()):
        return "qa"
    if metrics_path.name.startswith("metrics_"):
        if "bert" in metrics_path.stem or "static" in metrics_path.stem:
            return "qa"
    return "unknown"


def _infer_checkpoint_path(metrics_path: Path, task: str) -> Path | None:
    parent = metrics_path.parent
    if task == "qa":
        for name in ("bidaf_bert.pt", "bidaf_static.pt"):
            candidate = parent / name
            if candidate.exists():
                return candidate
    return None


def _build_legacy_record(metrics_path: Path) -> RunRecord:
    payload = load_json(metrics_path, default={}) or {}
    task = _infer_task_from_metrics(payload, metrics_path)
    checkpoint_path = _infer_checkpoint_path(metrics_path, task)

    if task == "sentiment":
        model_name = str(payload.get("model_name", "sentiment-model"))
        dataset = str(payload.get("dataset", "dataset"))
        label = f"{metrics_path.parent.name} | {model_name}"
        metadata = {
            "task": "sentiment",
            "model_name": model_name,
            "dataset": dataset,
            "metrics": payload,
        }
    elif task == "qa":
        embedding_mode = "bert" if "bert" in metrics_path.stem else "static"
        label = f"{metrics_path.parent.name} | {embedding_mode}"
        metadata = {
            "task": "qa",
            "embedding_mode": embedding_mode,
            "metrics": payload,
        }
    else:
        label = metrics_path.parent.name
        metadata = {
            "task": "unknown",
            "metrics": payload,
        }

    return RunRecord(
        task=task,
        label=label,
        output_dir=metrics_path.parent.resolve(),
        manifest_path=None,
        metrics_path=metrics_path.resolve(),
        checkpoint_path=checkpoint_path.resolve() if checkpoint_path is not None else None,
        metadata=metadata,
    )


def _build_manifest_record(manifest_path: Path) -> RunRecord | None:
    payload = load_json(manifest_path, default={}) or {}
    if not isinstance(payload, dict):
        return None

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return None

    task = payload.get("task")
    if not task:
        return None

    output_dir_value = payload.get("output_dir")
    if not output_dir_value:
        return None

    task = str(payload.get("task", "unknown"))
    output_dir = Path(output_dir_value).resolve()
    metrics_path = artifacts.get("metrics_path")
    checkpoint_path = artifacts.get("checkpoint_path")
    label = str(payload.get("label") or payload.get("run_name") or output_dir.name)
    return RunRecord(
        task=task,
        label=label,
        output_dir=output_dir,
        manifest_path=manifest_path.resolve(),
        metrics_path=Path(metrics_path).resolve() if metrics_path else None,
        checkpoint_path=Path(checkpoint_path).resolve() if checkpoint_path else None,
        metadata=payload,
    )


def discover_run_records(root: Path | None = None) -> list[RunRecord]:
    search_root = (root or project_root()).resolve()
    manifest_records: dict[Path, RunRecord] = {}
    metric_records: dict[Path, RunRecord] = {}

    for manifest_path in search_root.rglob("run_manifest.json"):
        if is_ignored_path(manifest_path):
            continue
        record = _build_manifest_record(manifest_path)
        if record is None:
            continue
        manifest_records[record.output_dir] = record

    for metrics_path in search_root.rglob("metrics*.json"):
        if is_ignored_path(metrics_path):
            continue
        output_dir = metrics_path.parent.resolve()
        if output_dir in manifest_records:
            continue
        metric_records[output_dir] = _build_legacy_record(metrics_path)

    all_records = list(manifest_records.values()) + list(metric_records.values())
    return sorted(
        all_records,
        key=lambda record: record.output_dir.stat().st_mtime if record.output_dir.exists() else 0.0,
        reverse=True,
    )


def discover_runs_by_task(task: str, root: Path | None = None) -> list[RunRecord]:
    return [record for record in discover_run_records(root=root) if record.task == task]


def discover_saved_bundles(root: Path | None = None) -> list[dict[str, Any]]:
    bundle_root = (root or project_root()).resolve() / "UI" / "artifacts" / "model_store"
    if not bundle_root.exists():
        return []

    bundles: list[dict[str, Any]] = []
    for metadata_path in bundle_root.rglob("bundle_metadata.json"):
        if is_ignored_path(metadata_path):
            continue
        metadata = load_json(metadata_path, default={}) or {}
        bundle_dir = metadata_path.parent.resolve()
        bundles.append(
            {
                "label": metadata.get("label", bundle_dir.name),
                "namespace": metadata.get("namespace", bundle_dir.parent.name),
                "path": bundle_dir,
                "metadata": metadata,
            }
        )

    return sorted(
        bundles,
        key=lambda bundle: bundle["path"].stat().st_mtime if bundle["path"].exists() else 0.0,
        reverse=True,
    )
