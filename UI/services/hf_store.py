from __future__ import annotations

from pathlib import Path
from typing import Any

from UI.services.artifacts import project_root, write_json


def slugify_identifier(value: str) -> str:
    slug = value.strip().replace("\\", "__").replace("/", "__").replace(":", "_")
    slug = slug.replace(" ", "_")
    return slug or "model"


def model_store_root(root: Path | None = None) -> Path:
    return (root or project_root()).resolve() / "UI" / "artifacts" / "model_store"


def bundle_dir(namespace: str, source_name: str, root: Path | None = None) -> Path:
    return model_store_root(root=root) / namespace / slugify_identifier(source_name)


def bundle_config_exists(path: str | Path) -> bool:
    directory = Path(path)
    return (directory / "config.json").exists()


def resolve_load_source(
    source_name: str,
    namespace: str,
    root: Path | None = None,
    *,
    prefer_cached_bundle: bool = True,
) -> str:
    candidate_path = Path(source_name)
    if candidate_path.exists():
        return str(candidate_path.resolve())

    if prefer_cached_bundle:
        cached_bundle = bundle_dir(namespace=namespace, source_name=source_name, root=root)
        if bundle_config_exists(cached_bundle):
            return str(cached_bundle.resolve())

    return source_name


def persist_pretrained_bundle(
    model,
    tokenizer,
    source_name: str,
    namespace: str,
    root: Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
    target_name: str | None = None,
) -> Path:
    storage_name = target_name or source_name
    target_dir = bundle_dir(namespace=namespace, source_name=storage_name, root=root)
    target_dir.mkdir(parents=True, exist_ok=True)

    # On Windows, repeatedly rewriting an already-loaded safetensors bundle can
    # fail if another process still has a mapped view open. Reuse the cache once
    # the bundle is present instead of overwriting it on every UI prediction.
    if not bundle_config_exists(target_dir):
        try:
            model.save_pretrained(target_dir)
        except OSError:
            model.save_pretrained(target_dir, safe_serialization=False)

        if tokenizer is not None:
            tokenizer.save_pretrained(target_dir)

    metadata = {
        "label": source_name,
        "source_name": source_name,
        "target_name": storage_name,
        "namespace": namespace,
        "path": str(target_dir.resolve()),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    write_json(target_dir / "bundle_metadata.json", metadata)
    return target_dir
