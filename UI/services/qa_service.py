from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from UI.services.artifacts import load_json, project_root
from UI.services.hf_store import persist_pretrained_bundle, resolve_load_source
from qa_system.data import TOKEN_PATTERN, UNK_IDX
from qa_system.metrics import extract_answer_text, select_best_span
from qa_system.model import BiDAFQuestionAnswering


@dataclass(slots=True)
class LoadedQAModel:
    model: BiDAFQuestionAnswering
    device: torch.device
    checkpoint_path: Path
    embedding_mode: str
    context_max_length: int
    question_max_length: int
    max_answer_length: int
    lowercase_static: bool
    tokenizer: Any | None
    vocab: dict[str, int] | None
    metadata: dict[str, Any]


def _simple_tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def _load_vocab(vocab_path: Path) -> dict[str, int]:
    with vocab_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _prepare_static_inputs(
    context: str,
    question: str,
    vocab: dict[str, int],
    *,
    context_max_length: int,
    question_max_length: int,
    lowercase: bool,
    device: torch.device,
) -> dict[str, Any]:
    context_tokens = _simple_tokenize(context)[:context_max_length]
    question_tokens = _simple_tokenize(question)[:question_max_length]
    if not context_tokens or not question_tokens:
        raise ValueError("Both context and question must contain at least one token.")

    offsets = [(start, end) for _, start, end in context_tokens]
    context_ids = [
        vocab.get(token.lower() if lowercase else token, UNK_IDX)
        for token, _, _ in context_tokens
    ]
    question_ids = [
        vocab.get(token.lower() if lowercase else token, UNK_IDX)
        for token, _, _ in question_tokens
    ]

    context_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
    question_tensor = torch.tensor([question_ids], dtype=torch.long, device=device)
    context_mask = torch.ones_like(context_tensor, dtype=torch.bool)
    question_mask = torch.ones_like(question_tensor, dtype=torch.bool)

    return {
        "context_ids": context_tensor,
        "question_ids": question_tensor,
        "context_mask": context_mask,
        "question_mask": question_mask,
        "offsets": offsets,
    }


def _prepare_bert_inputs(
    context: str,
    question: str,
    tokenizer,
    *,
    context_max_length: int,
    question_max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    context_encoding = tokenizer(
        context,
        add_special_tokens=False,
        truncation=True,
        max_length=context_max_length,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    question_encoding = tokenizer(
        question,
        add_special_tokens=False,
        truncation=True,
        max_length=question_max_length,
        return_attention_mask=False,
        return_offsets_mapping=False,
    )

    context_ids = context_encoding.get("input_ids", [])
    question_ids = question_encoding.get("input_ids", [])
    if not context_ids or not question_ids:
        raise ValueError("Both context and question must contain at least one token after tokenization.")

    context_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
    question_tensor = torch.tensor([question_ids], dtype=torch.long, device=device)
    context_mask = torch.ones_like(context_tensor, dtype=torch.bool)
    question_mask = torch.ones_like(question_tensor, dtype=torch.bool)

    return {
        "context_ids": context_tensor,
        "question_ids": question_tensor,
        "context_mask": context_mask,
        "question_mask": question_mask,
        "offsets": [tuple(offset) for offset in context_encoding["offset_mapping"]],
    }


def load_qa_model(
    checkpoint_path: str | Path,
    *,
    root: Path | None = None,
    device: str | None = None,
) -> LoadedQAModel:
    current_root = (root or project_root()).resolve()
    checkpoint_file = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    model_kwargs = dict(checkpoint["model_kwargs"])
    training_args = checkpoint.get("training_args", {})
    run_manifest = load_json(checkpoint_file.parent / "run_manifest.json", default={}) or {}

    tokenizer = None
    vocab = None
    embedding_mode = str(model_kwargs["embedding_mode"])
    if embedding_mode == "bert":
        original_backbone_name = str(model_kwargs["bert_model_name"])
        manifest_artifacts = run_manifest.get("artifacts", {})
        local_backbone_dir = manifest_artifacts.get("local_backbone_dir")
        if local_backbone_dir and Path(local_backbone_dir).exists():
            model_kwargs["bert_model_name"] = local_backbone_dir
        else:
            model_kwargs["bert_model_name"] = resolve_load_source(
                str(model_kwargs["bert_model_name"]),
                namespace="qa_backbones",
                root=current_root,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_kwargs["bert_model_name"], use_fast=True)
    else:
        vocab_path = checkpoint_file.parent / "vocab_static.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Static QA checkpoint requires a vocab file: {vocab_path}")
        vocab = _load_vocab(vocab_path)

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = BiDAFQuestionAnswering(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch_device)
    model.eval()

    return LoadedQAModel(
        model=model,
        device=torch_device,
        checkpoint_path=checkpoint_file,
        embedding_mode=embedding_mode,
        context_max_length=int(training_args.get("context_max_length", 256)),
        question_max_length=int(training_args.get("question_max_length", 64)),
        max_answer_length=int(training_args.get("max_answer_length", 30)),
        lowercase_static=bool(training_args.get("lowercase_static", False)),
        tokenizer=tokenizer,
        vocab=vocab,
        metadata={
            "checkpoint": checkpoint,
            "training_args": training_args,
            "run_manifest": run_manifest,
            "original_backbone_name": original_backbone_name if embedding_mode == "bert" else None,
        },
    )


def predict_answer(
    loaded: LoadedQAModel,
    *,
    context: str,
    question: str,
) -> dict[str, Any]:
    if loaded.embedding_mode == "bert":
        batch = _prepare_bert_inputs(
            context,
            question,
            loaded.tokenizer,
            context_max_length=loaded.context_max_length,
            question_max_length=loaded.question_max_length,
            device=loaded.device,
        )
    else:
        if loaded.vocab is None:
            raise ValueError("Static QA inference requires a loaded vocabulary.")
        batch = _prepare_static_inputs(
            context,
            question,
            loaded.vocab,
            context_max_length=loaded.context_max_length,
            question_max_length=loaded.question_max_length,
            lowercase=loaded.lowercase_static,
            device=loaded.device,
        )

    with torch.inference_mode():
        start_logits, end_logits = loaded.model(
            context_ids=batch["context_ids"],
            context_mask=batch["context_mask"],
            question_ids=batch["question_ids"],
            question_mask=batch["question_mask"],
        )

    valid_length = int(batch["context_mask"][0].sum().item())
    start_index, end_index = select_best_span(
        start_logits[0].detach().cpu(),
        end_logits[0].detach().cpu(),
        valid_length=valid_length,
        max_answer_length=loaded.max_answer_length,
    )
    answer_text = extract_answer_text(context, batch["offsets"], start_index, end_index)
    return {
        "answer": answer_text,
        "start_index": int(start_index),
        "end_index": int(end_index),
        "context_tokens_used": valid_length,
        "question_tokens_used": int(batch["question_mask"][0].sum().item()),
        "checkpoint_path": str(loaded.checkpoint_path),
        "embedding_mode": loaded.embedding_mode,
    }


def summarize_qa_model(loaded: LoadedQAModel) -> dict[str, Any]:
    training_args = loaded.metadata.get("training_args", {})
    checkpoint = loaded.metadata.get("checkpoint", {})
    history = checkpoint.get("history", [])
    return {
        "checkpoint_path": str(loaded.checkpoint_path),
        "embedding_mode": loaded.embedding_mode,
        "context_max_length": loaded.context_max_length,
        "question_max_length": loaded.question_max_length,
        "max_answer_length": loaded.max_answer_length,
        "lowercase_static": loaded.lowercase_static,
        "device": str(loaded.device),
        "epochs_recorded": len(history),
        "training_args": training_args,
    }


def persist_qa_backbone(loaded: LoadedQAModel, *, root: Path | None = None) -> Path | None:
    if loaded.embedding_mode != "bert" or loaded.tokenizer is None or not hasattr(loaded.model, "bert"):
        return None

    source_name = str(
        loaded.metadata.get("original_backbone_name")
        or loaded.metadata.get("training_args", {}).get("bert_model_name")
        or "qa-backbone"
    )
    return persist_pretrained_bundle(
        loaded.model.bert,
        loaded.tokenizer,
        source_name=source_name,
        namespace="qa_backbones",
        root=root,
        extra_metadata={
            "task": "qa",
            "checkpoint_path": str(loaded.checkpoint_path),
        },
    )
