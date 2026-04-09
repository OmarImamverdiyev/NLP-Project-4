from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from UI.services.artifacts import write_json, write_run_manifest
from UI.services.hf_store import persist_pretrained_bundle, resolve_load_source
from evaluate_sentiment140_bert import classification_report, load_examples, sample_examples, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a BERT sentiment classifier on Sentiment140-style CSV data."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("Sentiment140_v2.csv"),
        help="Path to the CSV file with sentiment labels and text.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Base Hugging Face model name or local path used for fine-tuning.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_task1_train_sentiment"),
        help="Directory where training outputs and metrics will be written.",
    )
    parser.add_argument(
        "--bundle-name",
        type=str,
        default=None,
        help="Optional name for the reusable UI model bundle. Defaults to the output directory name.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="polarity",
        help="Column containing sentiment labels.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column containing raw text.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100000,
        help="Maximum number of rows to sample before splitting. Set to 0 or a negative value to use all rows.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of sampled rows used for training.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
        help="Fraction of sampled rows used for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of sampled rows used for testing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used for both training and evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Tokenizer truncation length used during training and evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling and training.",
    )
    parser.add_argument(
        "--strip-html-breaks",
        action="store_true",
        help="Replace HTML <br> tags with spaces before splitting and training.",
    )
    return parser.parse_args()


def validate_split_ratios(args: argparse.Namespace) -> None:
    total = args.train_ratio + args.dev_ratio + args.test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, but received {args.train_ratio} + {args.dev_ratio} + {args.test_ratio} = {total}."
        )


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def summarize_lengths(tokenizer, texts: list[str], max_length: int) -> dict[str, float | int]:
    lengths: list[int] = []
    for index in range(0, len(texts), 512):
        batch_texts = texts[index : index + 512]
        encoded = tokenizer(
            batch_texts,
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])

    if not lengths:
        return {
            "average_tokenized_length": 0.0,
            "max_observed_tokenized_length": 0,
            "rows_truncated": 0,
        }

    return {
        "average_tokenized_length": round(safe_divide(sum(lengths), len(lengths)), 2),
        "max_observed_tokenized_length": max(lengths),
        "rows_truncated": sum(1 for length in lengths if length > max_length),
    }


def stratified_split(
    examples: list[tuple[int, str]],
    *,
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]], list[tuple[int, str]]]:
    generator = torch.Generator().manual_seed(seed)
    train_split: list[tuple[int, str]] = []
    dev_split: list[tuple[int, str]] = []
    test_split: list[tuple[int, str]] = []

    for label in (0, 1):
        label_examples = [example for example in examples if example[0] == label]
        if not label_examples:
            continue

        order = torch.randperm(len(label_examples), generator=generator).tolist()
        shuffled = [label_examples[index] for index in order]

        train_count = int(round(len(shuffled) * train_ratio))
        dev_count = int(round(len(shuffled) * dev_ratio))
        if train_count + dev_count > len(shuffled):
            dev_count = max(len(shuffled) - train_count, 0)
        test_count = len(shuffled) - train_count - dev_count

        train_split.extend(shuffled[:train_count])
        dev_split.extend(shuffled[train_count : train_count + dev_count])
        test_split.extend(shuffled[train_count + dev_count : train_count + dev_count + test_count])

    train_split = [train_split[index] for index in torch.randperm(len(train_split), generator=generator).tolist()]
    dev_split = [dev_split[index] for index in torch.randperm(len(dev_split), generator=generator).tolist()]
    test_split = [test_split[index] for index in torch.randperm(len(test_split), generator=generator).tolist()]
    return train_split, dev_split, test_split


@dataclass(slots=True)
class TextLabelDataset(Dataset):
    examples: list[tuple[int, str]]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[int, str]:
        return self.examples[index]


def build_dataloader(
    examples: list[tuple[int, str]],
    *,
    tokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TextLabelDataset(examples)

    def collate_fn(batch: list[tuple[int, str]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([label for label, _ in batch], dtype=torch.long)
        texts = [text for _, text in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        return encoded

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate_model(
    model,
    data_loader: DataLoader,
    *,
    device: torch.device,
    id2label: dict[int, str],
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    step_count = 0
    gold_labels: list[int] = []
    predicted_labels: list[int] = []
    sample_predictions: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch in data_loader:
            texts = batch.pop("texts", None)
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            step_count += 1

            probabilities = torch.softmax(outputs.logits.detach().cpu(), dim=-1)
            predictions = probabilities.argmax(dim=-1).tolist()
            labels = batch["labels"].detach().cpu().tolist()

            gold_labels.extend(labels)
            predicted_labels.extend(predictions)

            if texts is not None:
                for text, gold, pred, probs in zip(texts, labels, predictions, probabilities.tolist()):
                    if len(sample_predictions) >= 10:
                        break
                    sample_predictions.append(
                        {
                            "gold_label": int(gold),
                            "gold_label_name": id2label.get(int(gold), str(gold)),
                            "predicted_label": int(pred),
                            "predicted_label_name": id2label.get(int(pred), str(pred)),
                            "probabilities": {
                                id2label.get(index, str(index)): round(float(value), 6)
                                for index, value in enumerate(probs)
                            },
                            "text": text,
                        }
                    )

    metrics = classification_report(gold_labels, predicted_labels)
    average_loss = safe_divide(total_loss, step_count)
    return average_loss, metrics, sample_predictions


def attach_texts_to_loader(
    examples: list[tuple[int, str]],
    *,
    tokenizer,
    batch_size: int,
    max_length: int,
) -> DataLoader:
    dataset = TextLabelDataset(examples)

    def collate_fn(batch: list[tuple[int, str]]) -> dict[str, Any]:
        labels = torch.tensor([label for label, _ in batch], dtype=torch.long)
        texts = [text for _, text in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        encoded["texts"] = texts
        return encoded

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def main() -> None:
    args = parse_args()
    validate_split_ratios(args)
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    started = time.time()

    all_examples, class_counts = load_examples(
        args.csv_path,
        args.label_column,
        args.text_column,
        strip_html_breaks=args.strip_html_breaks,
    )
    sampled_examples = sample_examples(all_examples, args.max_rows, args.seed)
    train_examples, dev_examples, test_examples = stratified_split(
        sampled_examples,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    if not train_examples or not dev_examples or not test_examples:
        raise ValueError("Train/dev/test split produced an empty partition. Adjust max_rows or split ratios.")

    resolved_model_source = resolve_load_source(args.model_name, namespace="sentiment", root=project_root)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_source, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(resolved_model_source, num_labels=2)
    id2label = {0: "negative", 1: "positive"}
    label2id = {"negative": 0, "positive": 1}
    model.config.id2label = id2label
    model.config.label2id = label2id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = build_dataloader(
        train_examples,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
    )
    dev_loader = attach_texts_to_loader(
        dev_examples,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    test_loader = attach_texts_to_loader(
        test_examples,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, Any]] = []
    best_dev_macro_f1 = float("-inf")
    best_state_dict = None
    best_dev_metrics: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_train_loss += float(loss.item())
            train_steps += 1

        train_loss = safe_divide(total_train_loss, train_steps)
        dev_loss, dev_metrics, _ = evaluate_model(
            model,
            dev_loader,
            device=device,
            id2label=id2label,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_metrics.get("accuracy", 0.0),
            "dev_macro_f1": dev_metrics.get("macro_f1", 0.0),
        }
        history.append(epoch_result)
        print(
            "Epoch {epoch}: train_loss={train_loss:.4f} dev_loss={dev_loss:.4f} "
            "dev_accuracy={dev_accuracy:.4f} dev_macro_f1={dev_macro_f1:.4f}".format(**epoch_result)
        )

        current_dev_macro_f1 = float(dev_metrics.get("macro_f1", 0.0))
        if current_dev_macro_f1 >= best_dev_macro_f1:
            best_dev_macro_f1 = current_dev_macro_f1
            best_dev_metrics = dev_metrics
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state_dict is None:
        raise RuntimeError("Training finished without producing a best checkpoint.")

    model.load_state_dict(best_state_dict)
    model.to(device)
    test_loss, test_metrics, sample_predictions = evaluate_model(
        model,
        test_loader,
        device=device,
        id2label=id2label,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = args.bundle_name or args.output_dir.name
    local_model_dir = persist_pretrained_bundle(
        model,
        tokenizer,
        source_name=bundle_name,
        namespace="sentiment",
        root=project_root,
        extra_metadata={
            "task": "sentiment",
            "saved_from_script": "train_sentiment140_bert.py",
            "base_model_name": args.model_name,
            "resolved_source": resolved_model_source,
        },
    )

    training_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    dataset_summary = {
        "total_valid_rows": len(all_examples),
        **class_counts,
        "sampled_rows": len(sampled_examples),
        "train_rows": len(train_examples),
        "dev_rows": len(dev_examples),
        "test_rows": len(test_examples),
    }
    model_summary = {
        "num_labels": int(model.config.num_labels),
        "id2label": {str(key): value for key, value in id2label.items()},
        "max_length_used_for_eval": args.max_length,
        "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
        "tokenizer_do_lower_case": getattr(tokenizer, "do_lower_case", None),
        "base_model_name": args.model_name,
    }
    results = {
        "model_name": bundle_name,
        "base_model_name": args.model_name,
        "resolved_model_source": resolved_model_source,
        "dataset": str(args.csv_path),
        "trained_at_unix": int(time.time()),
        "runtime_seconds": round(time.time() - started, 2),
        "device": str(device),
        "label_mapping": {
            "dataset_negative": ["0", "negative"],
            "dataset_positive": ["1", "4", "positive"],
            "normalized_negative": 0,
            "normalized_positive": 1,
        },
        "dataset_summary": dataset_summary,
        "sample_summary": {
            "evaluated_rows": len(test_examples),
            "negative_rows": sum(1 for label, _ in test_examples if label == 0),
            "positive_rows": sum(1 for label, _ in test_examples if label == 1),
            "sampling_seed": args.seed,
            "balanced_sampling": args.max_rows > 0 and args.max_rows < len(all_examples),
        },
        "split_summary": {
            "train_ratio": args.train_ratio,
            "dev_ratio": args.dev_ratio,
            "test_ratio": args.test_ratio,
        },
        "model_summary": model_summary,
        "training_summary": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "best_dev_macro_f1": best_dev_macro_f1,
            "test_loss": test_loss,
        },
        "tokenization_summary": {
            "train": summarize_lengths(tokenizer, [text for _, text in train_examples], args.max_length),
            "dev": summarize_lengths(tokenizer, [text for _, text in dev_examples], args.max_length),
            "test": summarize_lengths(tokenizer, [text for _, text in test_examples], args.max_length),
        },
        "history": history,
        "metrics": test_metrics,
        "dev_metrics": best_dev_metrics,
        "sample_predictions": sample_predictions,
        "local_model_dir": str(local_model_dir.resolve()),
    }

    metrics_slug = bundle_name.replace("/", "__")
    metrics_path = args.output_dir / f"metrics_{metrics_slug}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    write_json(args.output_dir / "run_config.json", training_args)
    write_run_manifest(
        args.output_dir,
        {
            "task": "sentiment",
            "label": f"Sentiment | {bundle_name}",
            "run_name": args.output_dir.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": args.output_dir,
            "artifacts": {
                "metrics_path": metrics_path,
                "local_model_dir": local_model_dir,
            },
            "model": {
                "source_name": bundle_name,
                "base_model_name": args.model_name,
                "resolved_source": resolved_model_source,
                "local_model_dir": local_model_dir,
                "num_labels": int(model.config.num_labels),
                "id2label": {str(key): value for key, value in id2label.items()},
                "tokenizer_do_lower_case": getattr(tokenizer, "do_lower_case", None),
            },
            "dataset": {
                "path": args.csv_path,
                "label_column": args.label_column,
                "text_column": args.text_column,
                "max_rows": args.max_rows,
                "train_ratio": args.train_ratio,
                "dev_ratio": args.dev_ratio,
                "test_ratio": args.test_ratio,
            },
            "runtime": {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "max_length": args.max_length,
                "seed": args.seed,
                "device": str(device),
                "runtime_seconds": round(time.time() - started, 2),
            },
        },
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved reusable local bundle to: {local_model_dir}")


if __name__ == "__main__":
    main()
