from __future__ import annotations

import argparse
import csv
import re
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from UI.services.artifacts import write_json, write_run_manifest
from UI.services.hf_store import persist_pretrained_bundle, resolve_load_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned BERT sentiment classifier on Sentiment140-style CSV data."
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
        default="textattack/bert-base-uncased-SST-2",
        help="Hugging Face model name for a fine-tuned BERT sentiment classifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_task1_sentiment"),
        help="Directory where the metrics JSON will be written.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="polarity",
        help="Column containing gold sentiment labels.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column containing raw text.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=2000,
        help="Maximum number of rows to evaluate. Set to 0 or a negative value to use all rows.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenizer truncation length used during evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--strip-html-breaks",
        action="store_true",
        help="Replace HTML <br> tags with spaces before evaluation.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(raw_label: str) -> int | None:
    value = raw_label.strip().lower()
    if value in {"0", "negative", "neg"}:
        return 0
    if value in {"1", "4", "positive", "pos"}:
        return 1
    return None


HTML_BREAK_PATTERN = re.compile(r"(?:<br\s*/?>)+", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(raw_text: str, *, strip_html_breaks: bool) -> str:
    text = raw_text.strip()
    if strip_html_breaks:
        text = HTML_BREAK_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def load_examples(
    csv_path: Path,
    label_column: str,
    text_column: str,
    *,
    strip_html_breaks: bool,
) -> tuple[list[tuple[int, str]], dict[str, int]]:
    examples: list[tuple[int, str]] = []
    skipped = 0

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = normalize_label(row.get(label_column, ""))
            text = normalize_text(
                row.get(text_column, "") or "",
                strip_html_breaks=strip_html_breaks,
            )
            if label is None or not text:
                skipped += 1
                continue
            examples.append((label, text))

    class_counts = {
        "negative_rows": sum(1 for label, _ in examples if label == 0),
        "positive_rows": sum(1 for label, _ in examples if label == 1),
        "skipped_rows": skipped,
    }
    return examples, class_counts


def sample_examples(
    examples: list[tuple[int, str]],
    max_rows: int,
    seed: int,
) -> list[tuple[int, str]]:
    if max_rows <= 0 or max_rows >= len(examples):
        return list(examples)

    negatives = [example for example in examples if example[0] == 0]
    positives = [example for example in examples if example[0] == 1]
    rng = random.Random(seed)
    rng.shuffle(negatives)
    rng.shuffle(positives)

    target_neg = min(len(negatives), max_rows // 2)
    target_pos = min(len(positives), max_rows // 2)
    remaining = max_rows - target_neg - target_pos

    sample = negatives[:target_neg] + positives[:target_pos]

    if remaining > 0:
        leftovers = negatives[target_neg:] + positives[target_pos:]
        if leftovers:
            sample.extend(rng.sample(leftovers, min(remaining, len(leftovers))))

    rng.shuffle(sample)
    return sample


def batch_iterable(items: list[tuple[int, str]], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def classification_report(gold_labels: list[int], predicted_labels: list[int]) -> dict[str, object]:
    tp = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == 1 and pred == 1)
    tn = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == 0 and pred == 0)
    fp = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == 0 and pred == 1)
    fn = sum(1 for gold, pred in zip(gold_labels, predicted_labels) if gold == 1 and pred == 0)

    pos_precision = safe_divide(tp, tp + fp)
    pos_recall = safe_divide(tp, tp + fn)
    pos_f1 = safe_divide(2 * pos_precision * pos_recall, pos_precision + pos_recall)

    neg_precision = safe_divide(tn, tn + fn)
    neg_recall = safe_divide(tn, tn + fp)
    neg_f1 = safe_divide(2 * neg_precision * neg_recall, neg_precision + neg_recall)

    accuracy = safe_divide(tp + tn, len(gold_labels))
    macro_f1 = 0.5 * (pos_f1 + neg_f1)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "positive": {
            "precision": pos_precision,
            "recall": pos_recall,
            "f1": pos_f1,
            "support": sum(1 for label in gold_labels if label == 1),
        },
        "negative": {
            "precision": neg_precision,
            "recall": neg_recall,
            "f1": neg_f1,
            "support": sum(1 for label in gold_labels if label == 0),
        },
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }


def main() -> None:
    args = parse_args()
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

    resolved_model_source = resolve_load_source(args.model_name, namespace="sentiment", root=project_root)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_source, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(resolved_model_source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    gold_labels: list[int] = []
    predicted_labels: list[int] = []
    truncation_lengths: list[int] = []
    sample_predictions: list[dict[str, object]] = []

    with torch.inference_mode():
        for batch in batch_iterable(sampled_examples, args.batch_size):
            batch_gold = [label for label, _ in batch]
            batch_texts = [text for _, text in batch]

            untruncated = tokenizer(
                batch_texts,
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )
            truncation_lengths.extend(len(ids) for ids in untruncated["input_ids"])

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            logits = model(**encoded).logits
            batch_predictions = logits.argmax(dim=-1).detach().cpu().tolist()

            gold_labels.extend(batch_gold)
            predicted_labels.extend(batch_predictions)

            for gold, pred, text in zip(batch_gold, batch_predictions, batch_texts):
                if len(sample_predictions) >= 10:
                    break
                sample_predictions.append(
                    {
                        "gold_label": int(gold),
                        "predicted_label": int(pred),
                        "predicted_label_name": model.config.id2label.get(int(pred), str(pred)),
                        "text": text,
                    }
                )

    metrics = classification_report(gold_labels, predicted_labels)
    truncated_rows = sum(1 for length in truncation_lengths if length > args.max_length)
    avg_token_length = safe_divide(sum(truncation_lengths), len(truncation_lengths))

    tokenizer_model_max_length = getattr(tokenizer, "model_max_length", None)
    tokenizer_lowercases = getattr(tokenizer, "do_lower_case", None)

    results = {
        "model_name": args.model_name,
        "resolved_model_source": resolved_model_source,
        "dataset": str(args.csv_path),
        "evaluated_at_unix": int(time.time()),
        "runtime_seconds": round(time.time() - started, 2),
        "device": str(device),
        "label_mapping": {
            "dataset_negative": ["0", "negative"],
            "dataset_positive": ["1", "4", "positive"],
            "normalized_negative": 0,
            "normalized_positive": 1,
        },
        "dataset_summary": {
            "total_valid_rows": len(all_examples),
            **class_counts,
        },
        "sample_summary": {
            "evaluated_rows": len(sampled_examples),
            "negative_rows": sum(1 for label, _ in sampled_examples if label == 0),
            "positive_rows": sum(1 for label, _ in sampled_examples if label == 1),
            "sampling_seed": args.seed,
            "balanced_sampling": args.max_rows > 0 and args.max_rows < len(all_examples),
        },
        "model_summary": {
            "num_labels": int(model.config.num_labels),
            "id2label": {str(key): value for key, value in model.config.id2label.items()},
            "max_length_used_for_eval": args.max_length,
            "tokenizer_model_max_length": tokenizer_model_max_length,
            "tokenizer_do_lower_case": tokenizer_lowercases,
            "strip_html_breaks": args.strip_html_breaks,
        },
        "tokenization_summary": {
            "average_tokenized_length": round(avg_token_length, 2),
            "max_observed_tokenized_length": max(truncation_lengths) if truncation_lengths else 0,
            "rows_truncated_at_eval_time": truncated_rows,
        },
        "metrics": metrics,
        "sample_predictions": sample_predictions,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_model_dir = persist_pretrained_bundle(
        model,
        tokenizer,
        source_name=args.model_name,
        namespace="sentiment",
        root=project_root,
        extra_metadata={
            "task": "sentiment",
            "saved_from_script": "evaluate_sentiment140_bert.py",
        },
    )
    results["local_model_dir"] = str(local_model_dir.resolve())

    model_slug = args.model_name.replace("/", "__")
    metrics_path = args.output_dir / f"metrics_{model_slug}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    run_config = {
        "csv_path": args.csv_path,
        "model_name": args.model_name,
        "resolved_model_source": resolved_model_source,
        "output_dir": args.output_dir,
        "label_column": args.label_column,
        "text_column": args.text_column,
        "batch_size": args.batch_size,
        "max_rows": args.max_rows,
        "max_length": args.max_length,
        "seed": args.seed,
    }
    write_json(args.output_dir / "run_config.json", run_config)
    write_run_manifest(
        args.output_dir,
        {
            "task": "sentiment",
            "label": f"Sentiment | {args.model_name}",
            "run_name": args.output_dir.name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": args.output_dir,
            "artifacts": {
                "metrics_path": metrics_path,
                "local_model_dir": local_model_dir,
            },
            "model": {
                "source_name": args.model_name,
                "resolved_source": resolved_model_source,
                "local_model_dir": local_model_dir,
                "num_labels": int(model.config.num_labels),
                "id2label": {str(key): value for key, value in model.config.id2label.items()},
                "tokenizer_do_lower_case": tokenizer_lowercases,
            },
            "dataset": {
                "path": args.csv_path,
                "label_column": args.label_column,
                "text_column": args.text_column,
                "max_rows": args.max_rows,
            },
            "runtime": {
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "seed": args.seed,
                "device": str(device),
                "runtime_seconds": round(time.time() - started, 2),
            },
        },
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
