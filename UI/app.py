from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from UI.services.artifacts import (
    RunRecord,
    discover_run_records,
    discover_runs_by_task,
    discover_saved_bundles,
    load_json,
    load_presets,
    try_relative_path,
)
from UI.services.commands import build_qa_command, build_sentiment_command
from UI.services.qa_service import load_qa_model, persist_qa_backbone, predict_answer, summarize_qa_model
from UI.services.sentiment_service import (
    load_sentiment_model,
    persist_loaded_sentiment_model,
    predict_sentiment,
    summarize_sentiment_model,
)


st.set_page_config(page_title="NLP Project Dashboard", layout="wide")


@st.cache_resource(show_spinner=False)
def cached_sentiment_model(source_name: str):
    return load_sentiment_model(source_name, root=PROJECT_ROOT)


@st.cache_resource(show_spinner=False)
def cached_qa_model(checkpoint_path: str):
    return load_qa_model(checkpoint_path, root=PROJECT_ROOT)


def run_records() -> list[RunRecord]:
    return discover_run_records(root=PROJECT_ROOT)


def sentiment_runs() -> list[RunRecord]:
    return discover_runs_by_task("sentiment", root=PROJECT_ROOT)


def qa_runs() -> list[RunRecord]:
    return discover_runs_by_task("qa", root=PROJECT_ROOT)


def presets() -> dict[str, Any]:
    return load_presets()


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def run_label(record: RunRecord) -> str:
    return f"{record.label} [{try_relative_path(record.output_dir, base=PROJECT_ROOT)}]"


def path_from_dataset_preset(options: list[dict[str, Any]], index: int) -> tuple[str, dict[str, Any] | None]:
    if index < 0 or index >= len(options):
        return "", None
    preset = options[index]
    return str(PROJECT_ROOT / preset["path"]), preset


def sentiment_source_options() -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    seen_values: set[str] = set()

    for preset in presets().get("sentiment_models", []):
        value = str(preset["value"])
        if value in seen_values:
            continue
        options.append({"label": f'{preset["label"]} ({value})', "value": value})
        seen_values.add(value)

    for record in sentiment_runs():
        metrics_payload = record.metadata.get("metrics") or (
            load_json(record.metrics_path, default={}) if record.metrics_path else {}
        )
        value = str(metrics_payload.get("model_name", ""))
        if not value or value in seen_values:
            continue
        options.append({"label": f"From saved run: {value}", "value": value})
        seen_values.add(value)

    for bundle in discover_saved_bundles(root=PROJECT_ROOT):
        if bundle.get("namespace") != "sentiment":
            continue
        value = str(bundle["path"])
        if value in seen_values:
            continue
        options.append({"label": f'Local bundle: {bundle["label"]}', "value": value})
        seen_values.add(value)

    return options


def qa_checkpoint_options() -> list[RunRecord]:
    return [record for record in qa_runs() if record.checkpoint_path is not None]


def bundle_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for bundle in discover_saved_bundles(root=PROJECT_ROOT):
        rows.append(
            {
                "label": str(bundle["label"]),
                "namespace": str(bundle["namespace"]),
                "path": try_relative_path(Path(bundle["path"]), base=PROJECT_ROOT),
            }
        )
    return rows


def render_overview() -> None:
    st.title("NLP Project Dashboard")
    st.caption("The UI stays lightweight until you explicitly test a model. Saved runs and cached bundles are discovered automatically.")

    all_runs = run_records()
    sentiment = sentiment_runs()
    qa = qa_runs()
    bundles = bundle_rows()

    col1, col2, col3 = st.columns(3)
    col1.metric("Saved runs", len(all_runs))
    col2.metric("Sentiment runs", len(sentiment))
    col3.metric("QA runs", len(qa))

    col4, col5 = st.columns(2)
    col4.metric("Cached bundles", len(bundles))
    col5.metric("QA checkpoints", len(qa_checkpoint_options()))

    if bundles:
        st.subheader("Cached model bundles")
        st.dataframe(bundles, use_container_width=True, hide_index=True)
    else:
        st.info("No local UI model bundles have been saved yet. They will appear here after the first model test or future training/evaluation run.")

    report_col1, report_col2 = st.columns(2)
    with report_col1:
        with st.expander("Task 1 report", expanded=False):
            st.markdown(read_text_if_exists(PROJECT_ROOT / "task1_sentiment_analysis.md") or "Report not found.")
    with report_col2:
        with st.expander("Task 2 report", expanded=False):
            st.markdown(read_text_if_exists(PROJECT_ROOT / "task2_reading_comprehension_report.md") or "Report not found.")


def render_sentiment_panel() -> None:
    st.header("Task 1: Sentiment Analysis")

    result_col, test_col = st.columns((1.1, 0.9))
    with result_col:
        st.subheader("Saved results")
        records = [record for record in sentiment_runs() if record.metrics_path is not None]
        if not records:
            st.info("No sentiment metrics were discovered yet.")
        else:
            selected = st.selectbox(
                "Choose a saved sentiment run",
                options=records,
                format_func=run_label,
                key="sentiment_run_select",
            )
            payload = selected.metadata.get("metrics") or (
                load_json(selected.metrics_path, default={}) if selected.metrics_path else {}
            )
            metrics = payload.get("metrics", {})
            dataset_summary = payload.get("dataset_summary", {})
            model_summary = payload.get("model_summary", {})

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f'{100.0 * float(metrics.get("accuracy", 0.0)):.2f}%')
            m2.metric("Macro F1", f'{100.0 * float(metrics.get("macro_f1", 0.0)):.2f}%')
            m3.metric("Rows evaluated", int(payload.get("sample_summary", {}).get("evaluated_rows", 0)))
            m4.metric("Classes", int(model_summary.get("num_labels", 0)))

            st.write("Model summary")
            st.json(
                {
                    "model_name": payload.get("model_name"),
                    "max_length_used_for_eval": model_summary.get("max_length_used_for_eval"),
                    "tokenizer_do_lower_case": model_summary.get("tokenizer_do_lower_case"),
                    "tokenizer_model_max_length": model_summary.get("tokenizer_model_max_length"),
                }
            )

            st.write("Dataset summary")
            st.json(dataset_summary)

            samples = payload.get("sample_predictions", [])
            if samples:
                st.write("Sample predictions")
                st.dataframe(samples, use_container_width=True, hide_index=True)

    with test_col:
        st.subheader("Manual model test")
        source_options = sentiment_source_options()
        option_labels = [item["label"] for item in source_options] + ["Custom model name or path"]
        selected_label = st.selectbox("Model source", option_labels, key="sentiment_source_select")
        if selected_label == "Custom model name or path":
            source_value = st.text_input("Custom model", value="textattack/bert-base-uncased-SST-2")
        else:
            source_value = next(item["value"] for item in source_options if item["label"] == selected_label)

        sentiment_text = st.text_area(
            "Texts to classify",
            value="I really enjoyed this model.\nThis result needs more work.",
            height=160,
            key="sentiment_text_input",
        )
        max_length = st.number_input("Max token length", min_value=8, max_value=512, value=128, step=8)

        if st.button("Run sentiment prediction", key="run_sentiment_prediction"):
            texts = [line for line in sentiment_text.splitlines() if line.strip()]
            if not texts:
                st.warning("Enter at least one non-empty text line.")
            else:
                with st.spinner("Loading sentiment model..."):
                    loaded = cached_sentiment_model(source_value)
                    local_bundle_dir = persist_loaded_sentiment_model(loaded, root=PROJECT_ROOT)
                    predictions = predict_sentiment(loaded, texts, max_length=int(max_length))
                    summary = summarize_sentiment_model(loaded)
                    summary["local_bundle_dir"] = str(local_bundle_dir)
                st.success("Prediction finished. The model bundle is now available for UI reuse.")
                st.json(summary)
                st.dataframe(predictions, use_container_width=True, hide_index=True)

    st.subheader("Evaluation command builder")
    dataset_options = presets().get("datasets", {}).get("sentiment", [])
    dataset_labels = [item["label"] for item in dataset_options] + ["Custom dataset path"]
    selected_dataset = st.selectbox("Dataset preset", dataset_labels, key="sentiment_dataset_select")
    if selected_dataset == "Custom dataset path":
        csv_path = st.text_input("CSV path", value=str(PROJECT_ROOT / "Sentiment140_v2.csv"))
        label_column = st.text_input("Label column", value="polarity")
        text_column = st.text_input("Text column", value="text")
    else:
        preset_index = dataset_labels.index(selected_dataset)
        csv_path, preset_payload = path_from_dataset_preset(dataset_options, preset_index)
        label_column = st.text_input("Label column", value=str(preset_payload.get("label_column", "polarity")))
        text_column = st.text_input("Text column", value=str(preset_payload.get("text_column", "text")))

    builder_col1, builder_col2, builder_col3, builder_col4 = st.columns(4)
    model_input = builder_col1.text_input("Model name", value=source_options[0]["value"] if source_options else "textattack/bert-base-uncased-SST-2")
    output_dir = builder_col2.text_input("Output directory", value="outputs_task1_ui_sentiment")
    batch_size = builder_col3.number_input("Batch size", min_value=1, max_value=512, value=32, step=1)
    max_rows = builder_col4.number_input("Max rows", min_value=0, max_value=1000000, value=2000, step=100)
    extra_col1, extra_col2 = st.columns(2)
    max_length_for_eval = extra_col1.number_input("Evaluation max length", min_value=8, max_value=512, value=128, step=8)
    seed = extra_col2.number_input("Sampling seed", min_value=0, max_value=999999, value=42, step=1)

    command = build_sentiment_command(
        csv_path=csv_path,
        model_name=model_input,
        output_dir=output_dir,
        label_column=label_column,
        text_column=text_column,
        batch_size=int(batch_size),
        max_rows=int(max_rows),
        max_length=int(max_length_for_eval),
        seed=int(seed),
        root=PROJECT_ROOT,
    )
    st.code(command, language="powershell")
    st.caption("Future runs of the evaluation script now write a `run_manifest.json` file and save a reusable local model bundle for the UI.")


def render_qa_panel() -> None:
    st.header("Task 2: Reading Comprehension")

    result_col, test_col = st.columns((1.1, 0.9))
    with result_col:
        st.subheader("Saved results")
        records = qa_runs()
        if not records:
            st.info("No QA metrics were discovered yet.")
        else:
            selected = st.selectbox(
                "Choose a saved QA run",
                options=records,
                format_func=run_label,
                key="qa_run_select",
            )
            payload = selected.metadata.get("metrics") or (
                load_json(selected.metrics_path, default={}) if selected.metrics_path else {}
            )
            history = payload.get("history", [])
            data_summary = payload.get("data_summary", {})
            latest = history[-1] if history else {}

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Dev EM", f'{float(latest.get("dev_exact_match", 0.0)):.2f}')
            q2.metric("Dev F1", f'{float(latest.get("dev_f1", 0.0)):.2f}')
            q3.metric("Epochs", len(history))
            q4.metric("Train features", int(data_summary.get("train_features", 0)))

            if history:
                st.write("Epoch history")
                st.dataframe(history, use_container_width=True, hide_index=True)

            st.write("Data summary")
            st.json(data_summary)

    with test_col:
        st.subheader("Manual QA test")
        checkpoint_records = qa_checkpoint_options()
        if not checkpoint_records:
            st.info("No QA checkpoints are available for interactive testing yet.")
        else:
            selected_checkpoint = st.selectbox(
                "Checkpoint",
                options=checkpoint_records,
                format_func=run_label,
                key="qa_checkpoint_select",
            )
            context = st.text_area(
                "Context passage",
                value="Baku is the capital and largest city of Azerbaijan. It is located on the Caspian Sea.",
                height=140,
                key="qa_context_input",
            )
            question = st.text_input("Question", value="What is the capital of Azerbaijan?", key="qa_question_input")

            if st.button("Predict answer span", key="run_qa_prediction"):
                if not context.strip() or not question.strip():
                    st.warning("Both context and question are required.")
                else:
                    with st.spinner("Loading QA checkpoint..."):
                        loaded = cached_qa_model(str(selected_checkpoint.checkpoint_path))
                        local_backbone_dir = persist_qa_backbone(loaded, root=PROJECT_ROOT)
                        answer = predict_answer(loaded, context=context, question=question)
                        summary = summarize_qa_model(loaded)
                        summary["local_backbone_dir"] = str(local_backbone_dir) if local_backbone_dir else None
                    st.success("QA prediction finished.")
                    st.json(summary)
                    st.write("Predicted answer")
                    st.code(answer["answer"] or "[empty answer]")
                    st.json(answer)

    st.subheader("Training command builder")
    dataset_presets = presets().get("datasets", {})
    train_presets = dataset_presets.get("qa_train", [])
    dev_presets = dataset_presets.get("qa_dev", [])

    mode_col1, mode_col2, mode_col3 = st.columns(3)
    embedding_mode = mode_col1.selectbox("Embedding mode", ["static", "bert"], key="qa_embedding_mode")
    output_dir = mode_col2.text_input("Output directory", value="outputs_task2_ui")
    download_squad = mode_col3.checkbox("Download SQuAD if missing", value=True)

    train_labels = [item["label"] for item in train_presets] + ["Custom train path"]
    dev_labels = [item["label"] for item in dev_presets] + ["Custom dev path"]
    selected_train = st.selectbox("Train dataset", train_labels, key="qa_train_dataset_select")
    selected_dev = st.selectbox("Dev dataset", dev_labels, key="qa_dev_dataset_select")

    if selected_train == "Custom train path":
        train_file = st.text_input("Train file", value=str(PROJECT_ROOT / "data" / "train-v1.1.json"))
    else:
        train_file, _ = path_from_dataset_preset(train_presets, train_labels.index(selected_train))

    if selected_dev == "Custom dev path":
        dev_file = st.text_input("Dev file", value=str(PROJECT_ROOT / "data" / "dev-v1.1.json"))
    else:
        dev_file, _ = path_from_dataset_preset(dev_presets, dev_labels.index(selected_dev))

    b1, b2, b3, b4 = st.columns(4)
    train_limit = b1.number_input("Train limit", min_value=0, max_value=1000000, value=2000, step=100)
    dev_limit = b2.number_input("Dev limit", min_value=0, max_value=1000000, value=500, step=50)
    batch_size = b3.number_input("Batch size", min_value=1, max_value=512, value=4 if embedding_mode == "bert" else 16, step=1)
    epochs = b4.number_input("Epochs", min_value=1, max_value=100, value=2, step=1)

    c1, c2, c3, c4 = st.columns(4)
    context_max_length = c1.number_input("Context max length", min_value=16, max_value=512, value=128, step=8)
    question_max_length = c2.number_input("Question max length", min_value=4, max_value=256, value=32, step=4)
    embedding_dim = c3.number_input("Embedding dim", min_value=32, max_value=1024, value=100 if embedding_mode == "static" else 64, step=4)
    hidden_size = c4.number_input("Hidden size", min_value=32, max_value=1024, value=100 if embedding_mode == "static" else 64, step=4)

    d1, d2, d3 = st.columns(3)
    learning_rate = d1.number_input("Learning rate", min_value=0.000001, max_value=1.0, value=0.002, format="%.6f")
    dropout = d2.number_input("Dropout", min_value=0.0, max_value=0.9, value=0.2, step=0.05)
    max_answer_length = d3.number_input("Max answer length", min_value=1, max_value=128, value=30, step=1)

    lowercase_static = False
    glove_path: str | None = None
    bert_model_name = "bert-base-uncased"
    freeze_bert = False

    if embedding_mode == "static":
        s1, s2 = st.columns(2)
        lowercase_static = s1.checkbox("Lowercase static tokens", value=True)
        glove_path_raw = s2.text_input("Optional GloVe path", value="")
        glove_path = glove_path_raw.strip() or None
    else:
        qa_backbones = presets().get("qa_backbones", [])
        backbone_labels = [item["label"] for item in qa_backbones] + ["Custom backbone"]
        selected_backbone = st.selectbox("BERT backbone", backbone_labels, key="qa_backbone_select")
        if selected_backbone == "Custom backbone":
            bert_model_name = st.text_input("Custom backbone name", value="bert-base-uncased")
        else:
            bert_model_name = next(item["value"] for item in qa_backbones if item["label"] == selected_backbone)
        freeze_bert = st.checkbox("Freeze BERT backbone", value=True)

    command = build_qa_command(
        embedding_mode=embedding_mode,
        output_dir=output_dir,
        data_dir=str(PROJECT_ROOT / "data"),
        train_file=train_file,
        dev_file=dev_file,
        download_squad=download_squad,
        train_limit=None if int(train_limit) == 0 else int(train_limit),
        dev_limit=None if int(dev_limit) == 0 else int(dev_limit),
        batch_size=int(batch_size),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        dropout=float(dropout),
        embedding_dim=int(embedding_dim),
        hidden_size=int(hidden_size),
        context_max_length=int(context_max_length),
        question_max_length=int(question_max_length),
        max_answer_length=int(max_answer_length),
        lowercase_static=lowercase_static,
        glove_path=glove_path,
        bert_model_name=bert_model_name,
        freeze_bert=freeze_bert,
        root=PROJECT_ROOT,
    )
    st.code(command, language="powershell")
    st.caption("Future QA runs save checkpoints, metrics, training arguments, and a reusable run manifest for the dashboard.")


def main() -> None:
    overview_tab, sentiment_tab, qa_tab = st.tabs(["Overview", "Sentiment", "Reading Comprehension"])
    with overview_tab:
        render_overview()
    with sentiment_tab:
        render_sentiment_panel()
    with qa_tab:
        render_qa_panel()


if __name__ == "__main__":
    main()
