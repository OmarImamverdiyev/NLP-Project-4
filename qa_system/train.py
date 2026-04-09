from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from UI.services.artifacts import write_json, write_run_manifest
from UI.services.hf_store import persist_pretrained_bundle, resolve_load_source
from qa_system.data import (
    PAD_IDX,
    QADataset,
    build_bert_features,
    build_static_features,
    build_vocab,
    download_squad,
    load_glove_subset,
    load_squad_examples,
    pad_qa_batch,
    resolve_bert_max_length,
)
from qa_system.metrics import compute_squad_metrics, extract_answer_text, select_best_span
from qa_system.model import BiDAFQuestionAnswering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BiDAF-style QA system with static or BERT embeddings.")
    parser.add_argument("--embedding-mode", choices=["static", "bert"], default="static")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--dev-file", type=Path, default=None)
    parser.add_argument("--download-squad", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--context-max-length", type=int, default=256)
    parser.add_argument("--question-max-length", type=int, default=64)
    parser.add_argument("--max-answer-length", type=int, default=30)
    parser.add_argument("--vocab-max-size", type=int, default=50000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--lowercase-static", action="store_true")
    parser.add_argument("--glove-path", type=Path, default=None)
    parser.add_argument("--bert-model-name", type=str, default="bert-base-uncased")
    parser.add_argument(
        "--bert-architecture",
        choices=["legacy_bidaf", "joint_qa_transformers"],
        default="joint_qa_transformers",
    )
    parser.add_argument(
        "--bert-layer-combination",
        choices=["first", "last", "sum_last_four"],
        default="last",
    )
    parser.add_argument("--freeze-bert", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_backbone_snapshot_name(
    output_dir: Path,
    source_name: str,
    created_at: datetime,
) -> str:
    source_label = Path(str(source_name)).name or str(source_name)
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{output_dir.name}__{timestamp}__{source_label}"


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def resolve_learning_rate(args: argparse.Namespace) -> float:
    if args.learning_rate is not None:
        return float(args.learning_rate)
    if args.embedding_mode == "bert":
        return 3e-5
    return 2e-3


def is_joint_bert_model(model: BiDAFQuestionAnswering) -> bool:
    return (
        model.embedding_mode == "bert"
        and getattr(model, "bert_architecture", "") == "joint_qa_transformers"
    )


def is_legacy_bert_model(model: BiDAFQuestionAnswering) -> bool:
    return (
        model.embedding_mode == "bert"
        and getattr(model, "bert_architecture", "") == "legacy_bidaf"
    )


def resolve_data_files(args: argparse.Namespace) -> tuple[Path, Path]:
    train_file = args.train_file or args.data_dir / "train-v1.1.json"
    dev_file = args.dev_file or args.data_dir / "dev-v1.1.json"

    if args.download_squad or not train_file.exists() or not dev_file.exists():
        train_file, dev_file = download_squad(args.data_dir)

    return train_file, dev_file


def discover_glove_candidates(project_root: Path, embedding_dim: int | None = None) -> list[Path]:
    glove_dir = project_root / "glove"
    if not glove_dir.exists():
        return []

    patterns = [f"*{embedding_dim}d.txt"] if embedding_dim is not None else ["*.txt"]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(glove_dir.glob(pattern)))
    return [path for path in candidates if path.is_file()]


def format_glove_suggestion(project_root: Path, embedding_dim: int) -> str:
    candidates = discover_glove_candidates(project_root, embedding_dim=embedding_dim)
    if not candidates:
        return ""

    relative_paths = []
    for candidate in candidates[:4]:
        try:
            relative_paths.append(str(candidate.relative_to(project_root)))
        except ValueError:
            relative_paths.append(str(candidate))

    example = relative_paths[0]
    available = ", ".join(relative_paths)
    return (
        f" Try `--glove-path {example}`."
        f" Available local candidates: {available}."
    )


def resolve_glove_path(glove_path: Path, project_root: Path, embedding_dim: int) -> Path:
    raw_path = str(glove_path)
    normalized_path = raw_path.replace("\\", "/").lower()
    suggestion = format_glove_suggestion(project_root, embedding_dim)

    if normalized_path.startswith("path/to/"):
        raise FileNotFoundError(
            f"`--glove-path` is still using the placeholder path '{glove_path}'."
            f" Replace it with the real location of your GloVe file.{suggestion}"
        )

    resolved_path = glove_path.expanduser()
    if not resolved_path.is_absolute():
        resolved_path = (project_root / resolved_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"GloVe file not found: '{glove_path}' (resolved to '{resolved_path}').{suggestion}"
        )

    return resolved_path


def build_manifest_model_kwargs(model_kwargs: dict[str, object]) -> dict[str, object]:
    serializable_model_kwargs = dict(model_kwargs)
    pretrained_embeddings = serializable_model_kwargs.get("pretrained_embeddings")
    if torch.is_tensor(pretrained_embeddings):
        serializable_model_kwargs["pretrained_embeddings"] = {
            "stored_in_checkpoint": True,
            "shape": list(pretrained_embeddings.shape),
        }
    return serializable_model_kwargs


def prepare_dataloaders(args: argparse.Namespace):
    project_root = Path(__file__).resolve().parents[1]
    train_file, dev_file = resolve_data_files(args)
    train_examples = load_squad_examples(train_file, limit=args.train_limit)
    dev_examples = load_squad_examples(dev_file, limit=args.dev_limit)

    if args.embedding_mode == "static":
        vocab = build_vocab(
            train_examples,
            lowercase=args.lowercase_static,
            min_freq=args.min_freq,
            max_size=args.vocab_max_size,
        )
        pretrained_embeddings = None
        glove_hit_count = 0
        if args.glove_path is not None:
            resolved_glove_path = resolve_glove_path(
                args.glove_path,
                project_root=project_root,
                embedding_dim=args.embedding_dim,
            )
            pretrained_embeddings, glove_hit_count = load_glove_subset(
                resolved_glove_path,
                vocab,
                args.embedding_dim,
            )

        train_features, skipped_train = build_static_features(
            train_examples,
            vocab,
            context_max_length=args.context_max_length,
            question_max_length=args.question_max_length,
            lowercase=args.lowercase_static,
        )
        dev_features, skipped_dev = build_static_features(
            dev_examples,
            vocab,
            context_max_length=args.context_max_length,
            question_max_length=args.question_max_length,
            lowercase=args.lowercase_static,
        )
        pad_id = PAD_IDX
        model_kwargs = {
            "embedding_mode": "static",
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "vocab_size": len(vocab),
            "pad_idx": pad_id,
            "pretrained_embeddings": pretrained_embeddings,
        }
        extra_state = {
            "vocab": vocab,
            "glove_hit_count": glove_hit_count,
            "glove_path": str(resolved_glove_path) if args.glove_path is not None else None,
        }
    else:
        resolved_bert_source = resolve_load_source(
            args.bert_model_name,
            namespace="qa_backbones",
            root=project_root,
            prefer_cached_bundle=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(resolved_bert_source, use_fast=True)
        bert_max_length = resolve_bert_max_length(
            context_max_length=args.context_max_length,
            question_max_length=args.question_max_length,
            tokenizer=tokenizer,
        )
        train_features, skipped_train = build_bert_features(
            train_examples,
            tokenizer,
            context_max_length=args.context_max_length,
            question_max_length=args.question_max_length,
            architecture=args.bert_architecture,
        )
        dev_features, skipped_dev = build_bert_features(
            dev_examples,
            tokenizer,
            context_max_length=args.context_max_length,
            question_max_length=args.question_max_length,
            architecture=args.bert_architecture,
        )
        pad_id = tokenizer.pad_token_id or 0
        model_kwargs = {
            "embedding_mode": "bert",
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "bert_model_name": args.bert_model_name,
            "freeze_bert": args.freeze_bert,
            "bert_architecture": args.bert_architecture,
            "bert_layer_combination": args.bert_layer_combination,
        }
        extra_state = {
            "tokenizer_name": args.bert_model_name,
            "resolved_bert_source": resolved_bert_source,
            "tokenizer": tokenizer,
            "bert_max_length": bert_max_length,
        }

    train_dataset = QADataset(train_features)
    dev_dataset = QADataset(dev_features)
    collate_fn = partial(pad_qa_batch, pad_id=pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    data_summary = {
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "train_features": len(train_features),
        "dev_features": len(dev_features),
        "skipped_train": skipped_train,
        "skipped_dev": skipped_dev,
    }
    return train_loader, dev_loader, model_kwargs, data_summary, extra_state


def compute_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_positions: torch.Tensor,
    end_positions: torch.Tensor,
) -> torch.Tensor:
    start_loss = F.cross_entropy(start_logits, start_positions)
    end_loss = F.cross_entropy(end_logits, end_positions)
    return 0.5 * (start_loss + end_loss)


def evaluate(
    model: BiDAFQuestionAnswering,
    loader: DataLoader,
    device: torch.device,
    max_answer_length: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    predictions: dict[str, str] = {}
    references: dict[str, list[str]] = {}

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            if is_joint_bert_model(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    start_positions=batch["start_positions"],
                    end_positions=batch["end_positions"],
                )
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                loss = outputs.loss
            elif is_legacy_bert_model(model):
                start_logits, end_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    context_token_mask=batch["context_token_mask"],
                    question_token_mask=batch["question_token_mask"],
                )
                loss = compute_loss(
                    start_logits,
                    end_logits,
                    batch["start_positions"],
                    batch["end_positions"],
                )
            else:
                start_logits, end_logits = model(
                    context_ids=batch["context_ids"],
                    context_mask=batch["context_mask"],
                    question_ids=batch["question_ids"],
                    question_mask=batch["question_mask"],
                )
                loss = compute_loss(
                    start_logits,
                    end_logits,
                    batch["start_positions"],
                    batch["end_positions"],
                )
            total_loss += float(loss.item())
            total_batches += 1

            for row, metadata in enumerate(batch["metadata"]):
                candidate_mask = None
                valid_length = None
                if is_joint_bert_model(model):
                    candidate_mask = batch["context_token_mask"][row].detach().cpu()
                elif is_legacy_bert_model(model):
                    valid_length = int(batch["context_token_mask"][row].sum().item())
                else:
                    valid_length = int(batch["context_mask"][row].sum().item())
                start_index, end_index = select_best_span(
                    start_logits[row].detach().cpu(),
                    end_logits[row].detach().cpu(),
                    valid_length=valid_length,
                    max_answer_length=max_answer_length,
                    candidate_mask=candidate_mask,
                )
                prediction = extract_answer_text(
                    metadata["context"],
                    metadata["context_offsets"],
                    start_index,
                    end_index,
                )
                predictions[metadata["qid"]] = prediction
                references[metadata["qid"]] = metadata["gold_answers"]

    metrics = compute_squad_metrics(predictions, references)
    metrics["loss"] = total_loss / max(total_batches, 1)
    return metrics


def train(args: argparse.Namespace) -> dict[str, object]:
    args.learning_rate = resolve_learning_rate(args)
    set_seed(args.seed)
    project_root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, dev_loader, model_kwargs, data_summary, extra_state = prepare_dataloaders(args)

    runtime_model_kwargs = dict(model_kwargs)
    if args.embedding_mode == "bert":
        runtime_model_kwargs["bert_model_name"] = str(extra_state.get("resolved_bert_source", args.bert_model_name))

    model = BiDAFQuestionAnswering(**runtime_model_kwargs).to(device)
    optimizer_kwargs = {"lr": args.learning_rate}
    if is_joint_bert_model(model):
        optimizer_kwargs["eps"] = 1e-8
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        **optimizer_kwargs,
    )
    scheduler = None
    if is_joint_bert_model(model):
        total_steps = max(len(train_loader) * args.epochs, 1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps,
        )

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_batches = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if is_joint_bert_model(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    start_positions=batch["start_positions"],
                    end_positions=batch["end_positions"],
                )
                loss = outputs.loss
            elif is_legacy_bert_model(model):
                start_logits, end_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    context_token_mask=batch["context_token_mask"],
                    question_token_mask=batch["question_token_mask"],
                )
                loss = compute_loss(
                    start_logits,
                    end_logits,
                    batch["start_positions"],
                    batch["end_positions"],
                )
            else:
                start_logits, end_logits = model(
                    context_ids=batch["context_ids"],
                    context_mask=batch["context_mask"],
                    question_ids=batch["question_ids"],
                    question_mask=batch["question_mask"],
                )
                loss = compute_loss(
                    start_logits,
                    end_logits,
                    batch["start_positions"],
                    batch["end_positions"],
                )
            loss.backward()
            if is_joint_bert_model(model):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += float(loss.item())
            running_batches += 1

        dev_metrics = evaluate(
            model,
            dev_loader,
            device=device,
            max_answer_length=args.max_answer_length,
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / max(running_batches, 1),
            "dev_loss": dev_metrics["loss"],
            "dev_exact_match": dev_metrics["exact_match"],
            "dev_f1": dev_metrics["f1"],
        }
        history.append(epoch_result)
        print(
            "Epoch {epoch}: train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} "
            "dev_EM={dev_exact_match:.2f} dev_F1={dev_f1:.2f}".format(**epoch_result)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serializable_extra_state = {
        key: value
        for key, value in extra_state.items()
        if key not in {"vocab", "tokenizer"}
    }
    training_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_created_at = datetime.now(timezone.utc)
    checkpoint_path = args.output_dir / f"bidaf_{args.embedding_mode}.pt"
    local_backbone_dir = None
    if args.embedding_mode == "bert":
        tokenizer = extra_state.get("tokenizer")
        local_backbone_dir = persist_pretrained_bundle(
            model.bert,
            tokenizer,
            source_name=args.bert_model_name,
            namespace="qa_backbones",
            root=project_root,
            target_name=build_backbone_snapshot_name(
                args.output_dir,
                args.bert_model_name,
                run_created_at,
            ),
            extra_metadata={
                "task": "qa",
                "saved_from_script": "qa_system.train",
                "run_name": args.output_dir.name,
                "created_at_utc": run_created_at.isoformat(),
                "resolved_source": str(extra_state.get("resolved_bert_source", args.bert_model_name)),
            },
        )
        serializable_extra_state["local_backbone_dir"] = str(local_backbone_dir.resolve())

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "history": history,
            "data_summary": data_summary,
            "extra_state": serializable_extra_state,
            "training_args": training_args,
        },
        checkpoint_path,
    )

    metrics_path = args.output_dir / f"metrics_{args.embedding_mode}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "history": history,
                "data_summary": data_summary,
                "extra_state": serializable_extra_state,
                "training_args": training_args,
                "device": str(device),
            },
            handle,
            indent=2,
        )

    if "vocab" in extra_state:
        vocab_path = args.output_dir / "vocab_static.json"
        with vocab_path.open("w", encoding="utf-8") as handle:
            json.dump(extra_state["vocab"], handle, ensure_ascii=False, indent=2)
    else:
        vocab_path = None

    write_json(args.output_dir / "run_config.json", training_args)
    manifest_model_kwargs = build_manifest_model_kwargs(model_kwargs)
    write_run_manifest(
        args.output_dir,
        {
            "task": "qa",
            "label": f"QA | {args.embedding_mode} | {args.output_dir.name}",
            "run_name": args.output_dir.name,
            "created_at_utc": run_created_at.isoformat(),
            "output_dir": args.output_dir,
            "artifacts": {
                "checkpoint_path": checkpoint_path,
                "metrics_path": metrics_path,
                "vocab_path": vocab_path,
                "local_backbone_dir": local_backbone_dir,
            },
            "model": {
                "embedding_mode": args.embedding_mode,
                "model_kwargs": manifest_model_kwargs,
                "resolved_bert_source": serializable_extra_state.get("resolved_bert_source"),
                "tokenizer_name": serializable_extra_state.get("tokenizer_name"),
            },
            "dataset": {
                "data_dir": args.data_dir,
                "train_file": args.train_file,
                "dev_file": args.dev_file,
                "download_squad": args.download_squad,
                "train_limit": args.train_limit,
                "dev_limit": args.dev_limit,
            },
            "training_args": training_args,
            "data_summary": data_summary,
            "runtime": {
                "device": str(device),
            },
        },
    )

    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "history": history,
        "data_summary": data_summary,
        "device": str(device),
        "run_manifest_path": str((args.output_dir / "run_manifest.json").resolve()),
    }


def main() -> None:
    args = parse_args()
    results = train(args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
