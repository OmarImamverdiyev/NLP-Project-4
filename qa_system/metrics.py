from __future__ import annotations

import re
import string
from collections import Counter

import torch


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punctuation(value: str) -> str:
        punctuation = set(string.punctuation)
        return "".join(character for character in value if character not in punctuation)

    return white_space_fix(remove_articles(remove_punctuation(text.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    return max(metric_fn(prediction, truth) for truth in ground_truths)


def compute_squad_metrics(predictions: dict[str, str], references: dict[str, list[str]]) -> dict[str, float]:
    total = len(references)
    if total == 0:
        return {"exact_match": 0.0, "f1": 0.0}

    exact_match = 0.0
    f1 = 0.0
    for qid, ground_truths in references.items():
        prediction = predictions.get(qid, "")
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    return {
        "exact_match": 100.0 * exact_match / total,
        "f1": 100.0 * f1 / total,
    }


def select_best_span(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    valid_length: int,
    max_answer_length: int,
) -> tuple[int, int]:
    best_start = 0
    best_end = 0
    best_score = float("-inf")

    for start_index in range(valid_length):
        last_end = min(valid_length, start_index + max_answer_length)
        end_slice = end_logits[start_index:last_end]
        if end_slice.numel() == 0:
            continue
        relative_end = int(torch.argmax(end_slice).item())
        end_index = start_index + relative_end
        score = float(start_logits[start_index].item() + end_logits[end_index].item())
        if score > best_score:
            best_score = score
            best_start = start_index
            best_end = end_index

    return best_start, best_end


def extract_answer_text(
    context: str,
    offsets: list[tuple[int, int]],
    start_index: int,
    end_index: int,
) -> str:
    if not offsets:
        return ""
    if start_index < 0 or end_index >= len(offsets) or start_index > end_index:
        return ""

    start_char = offsets[start_index][0]
    end_char = offsets[end_index][1]
    if start_char >= end_char:
        return ""
    return context[start_char:end_char].strip()
