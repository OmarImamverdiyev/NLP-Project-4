# Task 2: Reading Comprehension System with BiDAF and BERT

## Overview

This task required a reading-comprehension system that can answer a question from a given context passage. The implementation in this repository is located in `qa_system/` and is written in **PyTorch**.

Main files:

- `qa_system/data.py`
- `qa_system/model.py`
- `qa_system/metrics.py`
- `qa_system/train.py`

The system supports two embedding modes:

- `static`: standard learned word embeddings, optionally initialized from GloVe
- `bert`: contextual embeddings from pretrained `bert-base-uncased`

The output of the model is a predicted **start position** and **end position** of the answer span inside the context passage.

## 1. BiDAF implementation

### Inputs

The model takes:

- a **context passage**
- a **question**

During training and evaluation, the code builds:

- `context_ids`
- `question_ids`
- `context_mask`
- `question_mask`
- gold `start_positions`
- gold `end_positions`

### Output

The model returns:

- `start_logits`
- `end_logits`

These logits are converted into the final answer span by selecting the best valid `(start, end)` pair in the context.

### Implemented architecture

The implementation is a practical BiDAF-style QA model with these stages:

1. **Embedding layer**
2. **Highway network**
3. **Shared BiLSTM contextual encoder**
4. **BiDAF attention flow**
5. **Modeling BiLSTM**
6. **Output BiLSTM**
7. **Start/end span projections**

The architecture is implemented in `qa_system/model.py`.

### Important implementation note

This is a **BiDAF-style** implementation, not a line-by-line reproduction of the original paper. In particular:

- it keeps the core attention-flow idea
- it predicts start and end answer positions as required
- it omits the original character-CNN block to keep the project lighter and easier to run on CPU

That trade-off is reasonable for an assignment implementation.

## 2. BERT-Base integration

### How BERT is used

The project integrates a pretrained BERT model through the `bert` embedding mode:

- model name: `bert-base-uncased`
- tokenizer: Hugging Face `AutoTokenizer`
- encoder: Hugging Face `AutoModel`

For this mode:

- the context is tokenized with BERT
- the question is tokenized with BERT
- BERT produces contextual hidden states for both sequences
- those hidden states are projected into the BiDAF embedding size
- the projected embeddings are passed into the rest of the BiDAF pipeline

### Two supported operating modes

The code supports:

- `--freeze-bert`: BERT is used as a frozen contextual embedding extractor
- without `--freeze-bert`: BERT is fine-tuned jointly with the QA model

For CPU-friendly testing, `--freeze-bert` is the better choice.

## 3. How BERT embeddings compare to traditional embeddings

### Static embeddings

With `--embedding-mode static`, the model uses:

- a learned embedding layer
- optionally GloVe initialization if a GloVe file is provided

These embeddings are:

- cheaper to train
- faster at inference and training
- not context-dependent

### BERT embeddings

With `--embedding-mode bert`, the model uses contextual representations:

- the same word can receive different vectors in different contexts
- question wording and context wording are represented more richly
- ambiguity is handled better than with a single static vector per token

### Expected effect on performance

On a properly sized QA experiment, BERT embeddings usually improve:

- Exact Match (EM)
- F1-score

compared with plain static embeddings such as Word2Vec or GloVe, because QA depends heavily on context-sensitive meaning.

### What happened in the local smoke tests

In the tiny local smoke tests, BERT did **not** beat the static baseline:

- static dev F1: **13.33**
- BERT dev F1: **12.50**

This does **not** mean static embeddings are better overall. The comparison is not scientifically meaningful because:

- the subsets were extremely small
- the BERT smoke run used only `1` train feature after preprocessing
- both runs were designed only to verify that the pipeline works end-to-end

So the correct interpretation is:

- the project successfully supports both static and BERT embeddings
- the tiny verification runs are too small to measure the true advantage of BERT

## 4. Training and evaluation setup

### Dataset

The code uses **SQuAD v1.1**.

The training script can automatically download:

- `data/train-v1.1.json`
- `data/dev-v1.1.json`

### Metrics

The evaluation uses:

- **Exact Match (EM)**
- **F1-score**

These are standard extractive QA metrics.

### Training objective

The model is trained with cross-entropy loss on:

- the gold start position
- the gold end position

The final loss is the average of the start-loss and end-loss.

### Saved artifacts

The training script writes:

- model checkpoint
- metrics JSON
- vocabulary JSON for static mode

Examples:

- `outputs_verify_static/metrics_static.json`
- `outputs_verify_bert/metrics_bert.json`

## 5. Local verification runs

I re-ran both smoke tests locally on **April 7, 2026** so the report reflects fresh execution in this workspace.

### Static embedding smoke test

Command:

```bash
.venv\Scripts\python -m qa_system.train --embedding-mode static --train-limit 16 --dev-limit 8 --epochs 1 --batch-size 4 --context-max-length 96 --question-max-length 24 --lowercase-static --output-dir outputs_verify_static
```

Results:

- train examples: `16`
- dev examples: `8`
- train features: `12`
- dev features: `6`
- skipped train: `4`
- skipped dev: `2`
- dev EM: **0.00**
- dev F1: **13.33**

Saved files:

- `outputs_verify_static/bidaf_static.pt`
- `outputs_verify_static/metrics_static.json`
- `outputs_verify_static/vocab_static.json`

### BERT embedding smoke test

Command:

```bash
.venv\Scripts\python -m qa_system.train --embedding-mode bert --freeze-bert --train-limit 4 --dev-limit 2 --epochs 1 --batch-size 1 --context-max-length 64 --question-max-length 24 --embedding-dim 64 --hidden-size 64 --output-dir outputs_verify_bert
```

Results:

- train examples: `4`
- dev examples: `2`
- train features: `1`
- dev features: `2`
- skipped train: `3`
- skipped dev: `0`
- dev EM: **0.00**
- dev F1: **12.50**

Saved files:

- `outputs_verify_bert/bidaf_bert.pt`
- `outputs_verify_bert/metrics_bert.json`

## 6. Analysis of the results

### What the smoke tests prove

The local runs prove that:

- the BiDAF-style model compiles and trains
- the static embedding path works
- the BERT embedding path works
- evaluation produces EM and F1 metrics
- checkpoints and metrics files are saved correctly

### What the smoke tests do not prove

They do **not** prove final task-level quality, because:

- the datasets were drastically reduced
- the models trained for only one epoch
- CPU-friendly limits forced aggressive simplification

So these runs should be described as **verification runs**, not final benchmark runs.

## 7. Recommended larger experiment

To obtain a more meaningful comparison between static embeddings and BERT embeddings, the following runs are recommended.

### Static baseline

```bash
.venv\Scripts\python -m qa_system.train --embedding-mode static --download-squad --train-limit 2000 --dev-limit 500 --epochs 2 --batch-size 16 --lowercase-static
```

### BERT-enhanced version

```bash
.venv\Scripts\python -m qa_system.train --embedding-mode bert --download-squad --freeze-bert --train-limit 2000 --dev-limit 500 --epochs 2 --batch-size 4 --context-max-length 128 --question-max-length 32
```

Then compare:

- `dev_exact_match`
- `dev_f1`

from the generated metrics JSON files.

## 8. Final conclusion

Task 2 has been implemented in this repository as a working **BiDAF-style extractive QA system** with optional **BERT-Base contextual embeddings**.

The project satisfies the assignment requirements at an implementation level:

- BiDAF-style architecture implemented in PyTorch
- question and context used as inputs
- answer start and end positions predicted as outputs
- BERT-Base integrated for contextual embeddings
- EM and F1 evaluation implemented

The fresh local smoke tests confirm that both the static and BERT versions run successfully end to end. The current scores are only smoke-test numbers, so a larger SQuAD experiment is still needed for a meaningful performance comparison.
