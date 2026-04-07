from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

SQUAD_URLS = {
    "train-v1.1.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev-v1.1.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(slots=True)
class QAExample:
    qid: str
    context: str
    question: str
    primary_answer: str
    answer_start: int
    answer_end: int
    gold_answers: list[str]


@dataclass(slots=True)
class QAFeature:
    qid: str
    context: str
    gold_answers: list[str]
    context_ids: list[int]
    question_ids: list[int]
    context_offsets: list[tuple[int, int]]
    start_position: int
    end_position: int


def download_squad(data_dir: str | Path) -> tuple[Path, Path]:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    downloaded_paths: list[Path] = []
    for filename, url in SQUAD_URLS.items():
        target = data_path / filename
        if not target.exists():
            urllib.request.urlretrieve(url, target)
        downloaded_paths.append(target)

    return downloaded_paths[0], downloaded_paths[1]


def load_squad_examples(path: str | Path, limit: int | None = None) -> list[QAExample]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    examples: list[QAExample] = []
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                answers = qa.get("answers", [])
                if not answers:
                    continue
                primary = answers[0]
                primary_text = primary["text"]
                answer_start = int(primary["answer_start"])
                examples.append(
                    QAExample(
                        qid=qa["id"],
                        context=context,
                        question=qa["question"],
                        primary_answer=primary_text,
                        answer_start=answer_start,
                        answer_end=answer_start + len(primary_text),
                        gold_answers=[answer["text"] for answer in answers],
                    )
                )
                if limit is not None and len(examples) >= limit:
                    return examples
    return examples


def simple_tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def build_vocab(
    examples: Iterable[QAExample],
    lowercase: bool = True,
    min_freq: int = 1,
    max_size: int | None = None,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for example in examples:
        for token, _, _ in simple_tokenize(example.context):
            normalized = token.lower() if lowercase else token
            counter[normalized] += 1
        for token, _, _ in simple_tokenize(example.question):
            normalized = token.lower() if lowercase else token
            counter[normalized] += 1

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for token, count in counter.most_common():
        if count < min_freq:
            continue
        if token in vocab:
            continue
        if max_size is not None and len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


def find_answer_token_span(
    offsets: list[tuple[int, int]],
    answer_start: int,
    answer_end: int,
) -> tuple[int, int] | None:
    overlapping = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_start < answer_end and token_end > answer_start
    ]
    if not overlapping:
        return None
    return overlapping[0], overlapping[-1]


def build_static_features(
    examples: Iterable[QAExample],
    vocab: dict[str, int],
    context_max_length: int,
    question_max_length: int,
    lowercase: bool = True,
) -> tuple[list[QAFeature], int]:
    features: list[QAFeature] = []
    skipped = 0

    for example in examples:
        context_tokens = simple_tokenize(example.context)
        question_tokens = simple_tokenize(example.question)

        context_offsets = [(start, end) for _, start, end in context_tokens]
        answer_span = find_answer_token_span(context_offsets, example.answer_start, example.answer_end)
        if answer_span is None:
            skipped += 1
            continue

        start_position, end_position = answer_span
        if end_position >= context_max_length:
            skipped += 1
            continue

        context_tokens = context_tokens[:context_max_length]
        question_tokens = question_tokens[:question_max_length]
        context_offsets = context_offsets[:context_max_length]

        context_ids = [
            vocab.get((token.lower() if lowercase else token), UNK_IDX)
            for token, _, _ in context_tokens
        ]
        question_ids = [
            vocab.get((token.lower() if lowercase else token), UNK_IDX)
            for token, _, _ in question_tokens
        ]

        if not context_ids or not question_ids:
            skipped += 1
            continue

        features.append(
            QAFeature(
                qid=example.qid,
                context=example.context,
                gold_answers=example.gold_answers,
                context_ids=context_ids,
                question_ids=question_ids,
                context_offsets=context_offsets,
                start_position=start_position,
                end_position=end_position,
            )
        )

    return features, skipped


def build_bert_features(
    examples: Iterable[QAExample],
    tokenizer,
    context_max_length: int,
    question_max_length: int,
) -> tuple[list[QAFeature], int]:
    features: list[QAFeature] = []
    skipped = 0

    for example in examples:
        context_encoding = tokenizer(
            example.context,
            add_special_tokens=False,
            truncation=True,
            max_length=context_max_length,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        question_encoding = tokenizer(
            example.question,
            add_special_tokens=False,
            truncation=True,
            max_length=question_max_length,
            return_attention_mask=False,
            return_offsets_mapping=False,
        )

        context_offsets = [tuple(offset) for offset in context_encoding["offset_mapping"]]
        answer_span = find_answer_token_span(context_offsets, example.answer_start, example.answer_end)
        if answer_span is None:
            skipped += 1
            continue

        if not context_encoding["input_ids"] or not question_encoding["input_ids"]:
            skipped += 1
            continue

        features.append(
            QAFeature(
                qid=example.qid,
                context=example.context,
                gold_answers=example.gold_answers,
                context_ids=list(context_encoding["input_ids"]),
                question_ids=list(question_encoding["input_ids"]),
                context_offsets=context_offsets,
                start_position=answer_span[0],
                end_position=answer_span[1],
            )
        )

    return features, skipped


def load_glove_subset(
    glove_path: str | Path,
    vocab: dict[str, int],
    embedding_dim: int,
) -> tuple[torch.Tensor, int]:
    embeddings = torch.empty(len(vocab), embedding_dim).normal_(mean=0.0, std=0.05)
    embeddings[PAD_IDX].zero_()

    found = 0
    with Path(glove_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            token, values = parts[0], parts[1:]
            if token not in vocab or len(values) != embedding_dim:
                continue
            embeddings[vocab[token]] = torch.tensor([float(value) for value in values], dtype=torch.float32)
            found += 1

    return embeddings, found


class QADataset(Dataset):
    def __init__(self, features: list[QAFeature]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> QAFeature:
        return self.features[index]


def pad_qa_batch(batch: list[QAFeature], pad_id: int) -> dict[str, torch.Tensor | list[dict[str, object]]]:
    batch_size = len(batch)
    max_context_len = max(len(feature.context_ids) for feature in batch)
    max_question_len = max(len(feature.question_ids) for feature in batch)

    context_ids = torch.full((batch_size, max_context_len), pad_id, dtype=torch.long)
    question_ids = torch.full((batch_size, max_question_len), pad_id, dtype=torch.long)
    context_mask = torch.zeros((batch_size, max_context_len), dtype=torch.bool)
    question_mask = torch.zeros((batch_size, max_question_len), dtype=torch.bool)
    start_positions = torch.empty(batch_size, dtype=torch.long)
    end_positions = torch.empty(batch_size, dtype=torch.long)

    metadata: list[dict[str, object]] = []
    for row, feature in enumerate(batch):
        context_length = len(feature.context_ids)
        question_length = len(feature.question_ids)

        context_ids[row, :context_length] = torch.tensor(feature.context_ids, dtype=torch.long)
        question_ids[row, :question_length] = torch.tensor(feature.question_ids, dtype=torch.long)
        context_mask[row, :context_length] = True
        question_mask[row, :question_length] = True
        start_positions[row] = feature.start_position
        end_positions[row] = feature.end_position
        metadata.append(
            {
                "qid": feature.qid,
                "context": feature.context,
                "gold_answers": feature.gold_answers,
                "context_offsets": feature.context_offsets,
            }
        )

    return {
        "context_ids": context_ids,
        "question_ids": question_ids,
        "context_mask": context_mask,
        "question_mask": question_mask,
        "start_positions": start_positions,
        "end_positions": end_positions,
        "metadata": metadata,
    }
