from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from UI.services.artifacts import project_root
from UI.services.hf_store import persist_pretrained_bundle, resolve_load_source


@dataclass(slots=True)
class LoadedSentimentModel:
    model: AutoModelForSequenceClassification
    tokenizer: Any
    device: torch.device
    source_name: str
    resolved_source: str
    local_bundle_dir: Path | None


def load_sentiment_model(
    source_name: str,
    *,
    root: Path | None = None,
    namespace: str = "sentiment",
    device: str | None = None,
) -> LoadedSentimentModel:
    current_root = (root or project_root()).resolve()
    resolved_source = resolve_load_source(source_name, namespace=namespace, root=current_root)
    tokenizer = AutoTokenizer.from_pretrained(resolved_source, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(resolved_source)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(torch_device)
    model.eval()

    local_bundle_dir: Path | None = None
    resolved_path = Path(resolved_source)
    if resolved_path.exists():
        local_bundle_dir = resolved_path.resolve()

    return LoadedSentimentModel(
        model=model,
        tokenizer=tokenizer,
        device=torch_device,
        source_name=source_name,
        resolved_source=resolved_source,
        local_bundle_dir=local_bundle_dir,
    )


def persist_loaded_sentiment_model(
    loaded: LoadedSentimentModel,
    *,
    root: Path | None = None,
    namespace: str = "sentiment",
) -> Path:
    return persist_pretrained_bundle(
        loaded.model,
        loaded.tokenizer,
        source_name=loaded.source_name,
        namespace=namespace,
        root=root,
        extra_metadata={
            "task": "sentiment",
            "resolved_source": loaded.resolved_source,
        },
    )


def predict_sentiment(
    loaded: LoadedSentimentModel,
    texts: list[str],
    *,
    max_length: int = 128,
) -> list[dict[str, Any]]:
    clean_texts = [text.strip() for text in texts if text.strip()]
    if not clean_texts:
        return []

    encoded = loaded.tokenizer(
        clean_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(loaded.device) for key, value in encoded.items()}

    with torch.inference_mode():
        logits = loaded.model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).detach().cpu()
        predictions = probabilities.argmax(dim=-1).tolist()

    results: list[dict[str, Any]] = []
    for text, predicted_index, probs in zip(clean_texts, predictions, probabilities.tolist()):
        label_name = loaded.model.config.id2label.get(int(predicted_index), str(predicted_index))
        probability_map = {
            loaded.model.config.id2label.get(index, str(index)): round(float(value), 6)
            for index, value in enumerate(probs)
        }
        results.append(
            {
                "text": text,
                "predicted_index": int(predicted_index),
                "predicted_label": label_name,
                "probabilities": probability_map,
            }
        )

    return results


def summarize_sentiment_model(loaded: LoadedSentimentModel) -> dict[str, Any]:
    tokenizer = loaded.tokenizer
    config = loaded.model.config
    return {
        "source_name": loaded.source_name,
        "resolved_source": loaded.resolved_source,
        "local_bundle_dir": str(loaded.local_bundle_dir) if loaded.local_bundle_dir else None,
        "num_labels": int(config.num_labels),
        "id2label": {str(key): value for key, value in config.id2label.items()},
        "tokenizer_do_lower_case": getattr(tokenizer, "do_lower_case", None),
        "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
        "device": str(loaded.device),
    }
