from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL = "StartZer0/az-sentiment-bert"
LABEL_MAP = {
    0: "negative",
    1: "positive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Azerbaijani sentiment analysis with StartZer0/az-sentiment-bert."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Hugging Face model name or a local model directory.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="A single Azerbaijani text input. Pass multiple times for multiple texts.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional UTF-8 text file containing one example per line.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenizer truncation length.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("task1_extended") / "outputs" / "sample_predictions.json",
        help="JSON file where predictions will be saved.",
    )
    return parser.parse_args()


def load_texts(args: argparse.Namespace) -> list[str]:
    texts = [text.strip() for text in args.text if text.strip()]

    if args.input_file is not None:
        file_texts = [
            line.strip()
            for line in args.input_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        texts.extend(file_texts)

    if texts:
        return texts

    return [
        "Bu mehsul cox yaxsidir, cox razi qaldim.",
        "Xidmet berbad idi, bir daha istifade etmerem.",
        "Catdirilma cox suretli oldu ve keyfiyyet mene xos geldi.",
        "Proqram tez-tez donur ve islemir.",
    ]


def main() -> None:
    args = parse_args()
    texts = load_texts(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).detach().cpu()
        predictions = probabilities.argmax(dim=-1).tolist()

    results: list[dict[str, object]] = []
    for text, pred_idx, probs in zip(texts, predictions, probabilities.tolist()):
        result = {
            "text": text,
            "predicted_index": int(pred_idx),
            "predicted_sentiment": LABEL_MAP.get(int(pred_idx), f"label_{pred_idx}"),
            "probabilities": {
                "negative": round(float(probs[0]), 6),
                "positive": round(float(probs[1]), 6),
            },
        }
        results.append(result)

    payload = {
        "model_name": args.model_name,
        "device": str(device),
        "num_labels": int(model.config.num_labels),
        "id2label": {str(key): value for key, value in model.config.id2label.items()},
        "tokenizer_do_lower_case": getattr(tokenizer, "do_lower_case", None),
        "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
        "predictions": results,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved predictions to: {args.output_file}")


if __name__ == "__main__":
    main()
