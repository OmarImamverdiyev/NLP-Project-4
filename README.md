# NLP Project 5

Datasets: https://drive.google.com/drive/folders/1NYBJmfx9RETVmYFXmHjYObUJQNuz7fZB?usp=sharing 

# Task 1
.venv\Scripts\python evaluate_sentiment140_bert.py --csv-path Sentiment140_v2.csv --model-name textattack/bert-base-uncased-SST-2 --max-rows 100000 --batch-size 64 --max-length 64 --output-dir outputs_task1_sentiment_100k


# Task 2 - static
.venv\Scripts\python -m qa_system.train --embedding-mode static --download-squad --train-limit 4380 --dev-limit 634 --epochs 2 --batch-size 16 --lowercase-static --output-dir outputs_task2_static_5pct_6pct

# Task 2 - BERT
.venv\Scripts\python -m qa_system.train --embedding-mode bert --download-squad --freeze-bert --train-limit 4380 --dev-limit 634 --epochs 2 --batch-size 4 --context-max-length 128 --question-max-length 32 --output-dir outputs_task2_bert_5pct_6pct

This repository now contains two assignment deliverables:

- [Task 1 analysis](task1_sentiment_analysis.md): answers for the BERT sentiment-analysis questions
- [Task 2 report](task2_reading_comprehension_report.md): architecture notes, evaluation setup, and smoke-test results
- `qa_system/`: a PyTorch implementation of a BiDAF-style reading-comprehension system with either static word embeddings or contextual BERT embeddings

## Task 2 overview

The QA system follows the BiDAF pipeline at a practical level:

- question and context are encoded separately
- embeddings can come from a static embedding layer or a pretrained BERT model
- a shared BiLSTM contextual encoder builds token representations
- BiDAF attention flow mixes question and context information
- span heads predict answer start and end positions in the context

To keep the project runnable on CPU, this implementation focuses on the core BiDAF attention-and-span architecture and does not include the original character CNN block.

## Files

- `qa_system/data.py`: SQuAD download/loading, tokenization, feature building, batching
- `qa_system/model.py`: BiDAF-style QA model with static or BERT embeddings
- `qa_system/metrics.py`: Exact Match, F1, answer decoding
- `qa_system/train.py`: training and evaluation entry point

## Install

```bash
python -m venv .venv --system-site-packages
.venv\Scripts\python -m pip install -r requirements.txt
```

If your base Python already has `NumPy 2.x` plus older `pandas`/`pyarrow`/`scikit-learn` wheels, using the local `.venv` is the safest route.

You can also install directly into your active environment:

```bash
pip install -r requirements.txt
```

## Run with static embeddings

Small CPU-friendly run:

```bash
python -m qa_system.train --embedding-mode static --download-squad --train-limit 200 --dev-limit 50 --epochs 1 --batch-size 8 --lowercase-static
```

If you want a more traditional baseline, you can initialize the static embedding layer with GloVe:

```bash
python -m qa_system.train --embedding-mode static --download-squad --glove-path path/to/glove.6B.100d.txt --embedding-dim 100 --lowercase-static
```

## Run with BERT embeddings

Recommended first smoke test on CPU:

```bash
python -m qa_system.train --embedding-mode bert --download-squad --freeze-bert --train-limit 32 --dev-limit 16 --epochs 1 --batch-size 2 --context-max-length 128 --question-max-length 32
```

Notes:

- `--freeze-bert` uses BERT as a contextual embedding extractor, which is much lighter than full fine-tuning
- removing `--freeze-bert` fine-tunes BERT end to end, but it is significantly slower

## Output

Training writes:

- `outputs/bidaf_static.pt` or `outputs/bidaf_bert.pt`
- `outputs/metrics_static.json` or `outputs/metrics_bert.json`
- `outputs/vocab_static.json` for the static baseline

## Comparing static vs BERT embeddings

Expected behavior:

- static embeddings are faster and lighter
- BERT embeddings usually improve EM and F1 because they are contextual and handle ambiguity better
- the gain is usually strongest when the same word has different meanings depending on the question/context pair

In this codebase, you can compare them by running:

1. Static baseline
```bash
python -m qa_system.train --embedding-mode static --download-squad --train-limit 2000 --dev-limit 500 --epochs 2 --batch-size 16 --lowercase-static
```
2. BERT-enhanced version
```bash
python -m qa_system.train --embedding-mode bert --download-squad --freeze-bert --train-limit 2000 --dev-limit 500 --epochs 2 --batch-size 4 --context-max-length 128 --question-max-length 32
```

Then compare `dev_exact_match` and `dev_f1` in the saved metrics JSON files.

## Azerbaijani note

The Task 1 tutorial uses English `bert-base-uncased`, which is not a good production choice for Azerbaijani. For Azerbaijani sentiment or QA, use a multilingual model such as `bert-base-multilingual-cased` or XLM-R and fine-tune it on Azerbaijani data.
