from __future__ import annotations

import subprocess
from pathlib import Path

from UI.services.artifacts import project_root, try_relative_path


def default_python_command(root: Path | None = None) -> str:
    current_root = (root or project_root()).resolve()
    venv_python = current_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return try_relative_path(venv_python, base=current_root)
    return "python"


def windows_cli(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts)


def build_sentiment_command(
    *,
    csv_path: str,
    model_name: str,
    output_dir: str,
    label_column: str = "polarity",
    text_column: str = "text",
    batch_size: int = 32,
    max_rows: int = 2000,
    max_length: int = 128,
    seed: int = 42,
    root: Path | None = None,
) -> str:
    parts = [
        default_python_command(root=root),
        "evaluate_sentiment140_bert.py",
        "--csv-path",
        csv_path,
        "--model-name",
        model_name,
        "--output-dir",
        output_dir,
        "--label-column",
        label_column,
        "--text-column",
        text_column,
        "--batch-size",
        str(batch_size),
        "--max-rows",
        str(max_rows),
        "--max-length",
        str(max_length),
        "--seed",
        str(seed),
    ]
    return windows_cli(parts)


def build_qa_command(
    *,
    embedding_mode: str,
    output_dir: str,
    data_dir: str = "data",
    train_file: str | None = None,
    dev_file: str | None = None,
    download_squad: bool = False,
    train_limit: int | None = None,
    dev_limit: int | None = None,
    batch_size: int = 8,
    epochs: int = 2,
    learning_rate: float | None = None,
    dropout: float = 0.2,
    embedding_dim: int = 100,
    hidden_size: int = 100,
    context_max_length: int = 256,
    question_max_length: int = 64,
    max_answer_length: int = 30,
    vocab_max_size: int = 50000,
    min_freq: int = 1,
    lowercase_static: bool = False,
    glove_path: str | None = None,
    bert_model_name: str = "bert-base-uncased",
    freeze_bert: bool = False,
    seed: int = 42,
    root: Path | None = None,
) -> str:
    parts = [
        default_python_command(root=root),
        "-m",
        "qa_system.train",
        "--embedding-mode",
        embedding_mode,
        "--data-dir",
        data_dir,
        "--output-dir",
        output_dir,
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--dropout",
        str(dropout),
        "--embedding-dim",
        str(embedding_dim),
        "--hidden-size",
        str(hidden_size),
        "--context-max-length",
        str(context_max_length),
        "--question-max-length",
        str(question_max_length),
        "--max-answer-length",
        str(max_answer_length),
        "--vocab-max-size",
        str(vocab_max_size),
        "--min-freq",
        str(min_freq),
        "--seed",
        str(seed),
    ]

    if learning_rate is not None:
        parts.extend(["--learning-rate", str(learning_rate)])

    if train_file:
        parts.extend(["--train-file", train_file])
    if dev_file:
        parts.extend(["--dev-file", dev_file])
    if download_squad:
        parts.append("--download-squad")
    if train_limit is not None:
        parts.extend(["--train-limit", str(train_limit)])
    if dev_limit is not None:
        parts.extend(["--dev-limit", str(dev_limit)])

    if embedding_mode == "static":
        if lowercase_static:
            parts.append("--lowercase-static")
        if glove_path:
            parts.extend(["--glove-path", glove_path])
    else:
        parts.extend(["--bert-model-name", bert_model_name])
        if freeze_bert:
            parts.append("--freeze-bert")

    return windows_cli(parts)
