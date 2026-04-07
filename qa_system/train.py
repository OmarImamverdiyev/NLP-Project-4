from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

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
    parser.add_argument("--learning-rate", type=float, default=2e-3)
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
    parser.add_argument("--freeze-bert", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def resolve_data_files(args: argparse.Namespace) -> tuple[Path, Path]:
    train_file = args.train_file or args.data_dir / "train-v1.1.json"
    dev_file = args.dev_file or args.data_dir / "dev-v1.1.json"

    if args.download_squad or not train_file.exists() or not dev_file.exists():
        train_file, dev_file = download_squad(args.data_dir)

    return train_file, dev_file


def prepare_dataloaders(args: argparse.Namespace):
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
            pretrained_embeddings, glove_hit_count = load_glove_subset(
                args.glove_path,
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
        }
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, use_fast=True)
        train_features, skipped_train = build_bert_features(
            train_examples,
            tokenizer,
            context_max_length=args.context_max_length,
            question_max_length=args.question_max_length,
        )
        dev_features, skipped_dev = build_bert_features(
            dev_examples,
            tokenizer,
            context_max_length=args.context_max_length,
            question_max_length=args.question_max_length,
        )
        pad_id = tokenizer.pad_token_id or 0
        model_kwargs = {
            "embedding_mode": "bert",
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "bert_model_name": args.bert_model_name,
            "freeze_bert": args.freeze_bert,
        }
        extra_state = {
            "tokenizer_name": args.bert_model_name,
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
                valid_length = int(batch["context_mask"][row].sum().item())
                start_index, end_index = select_best_span(
                    start_logits[row].detach().cpu(),
                    end_logits[row].detach().cpu(),
                    valid_length=valid_length,
                    max_answer_length=max_answer_length,
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
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, dev_loader, model_kwargs, data_summary, extra_state = prepare_dataloaders(args)

    model = BiDAFQuestionAnswering(**model_kwargs).to(device)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_batches = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
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
            optimizer.step()

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
    serializable_extra_state = {key: value for key, value in extra_state.items() if key != "vocab"}
    checkpoint_path = args.output_dir / f"bidaf_{args.embedding_mode}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "history": history,
            "data_summary": data_summary,
            "extra_state": serializable_extra_state,
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
                "device": str(device),
            },
            handle,
            indent=2,
        )

    if "vocab" in extra_state:
        vocab_path = args.output_dir / "vocab_static.json"
        with vocab_path.open("w", encoding="utf-8") as handle:
            json.dump(extra_state["vocab"], handle, ensure_ascii=False, indent=2)

    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "history": history,
        "data_summary": data_summary,
        "device": str(device),
    }


def main() -> None:
    args = parse_args()
    results = train(args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
