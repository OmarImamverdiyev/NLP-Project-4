# Task 1: Sentiment Analysis Using BERT

## Model selected

For Task 1, I analyzed the open-source model `textattack/bert-base-uncased-SST-2`:

- Hugging Face model: https://huggingface.co/textattack/bert-base-uncased-SST-2
- Base architecture: `bert-base-uncased`
- Task type: sequence classification for sentiment

This is a practical choice for the assignment because it is already fine-tuned for binary sentiment classification, while still using the same `bert-base-uncased` family commonly shown in introductory BERT sentiment tutorials.

I also tested this model locally on `Sentiment140_v2.csv` and saved the results to:

- `outputs_task1_sentiment/metrics_textattack__bert-base-uncased-SST-2.json`

## 1. What are the inputs and outputs of this model?

### Inputs

The model receives tokenized text in the standard BERT sequence-classification format:

- `input_ids`: integer token IDs produced by the BERT tokenizer
- `attention_mask`: marks real tokens with `1` and padding with `0`
- `token_type_ids` (optional): segment IDs, mainly useful when two sequences are packed together

For a normal sentiment-classification example, the raw input is a single sentence, review, or tweet. The tokenizer converts it into:

`[CLS] text tokens [SEP]`

### Outputs

The model outputs:

- a logits vector of size `2` for each input text
- after `softmax`, these logits become class probabilities
- the predicted class is the label with the highest probability

For this checkpoint, the two labels correspond to binary sentiment:

- class `0`: negative
- class `1`: positive

## 2. How many classes does it have?

The model has **2 classes**.

This was also confirmed locally from the loaded model configuration:

- `num_labels = 2`

## 3. What is the size of the input?

There are two useful ways to interpret "input size":

### Maximum sequence length

The underlying BERT tokenizer supports sequences up to **512 tokens**.

That was confirmed locally during evaluation:

- `tokenizer_model_max_length = 512`

### Model size

Because this checkpoint is built on `bert-base-uncased`, it inherits the standard BERT-Base architecture:

- 12 transformer layers
- hidden size 768
- 12 attention heads
- about 110 million parameters

### Local Sentiment140 note

For the Sentiment140 test, I used `max_length = 64` because tweets are short. On the 2,000-row sample:

- average tokenized length: **22.77**
- max observed tokenized length: **68**
- truncated rows at evaluation time: **1**

So the model itself supports 512 tokens, but the local test used 64-token truncation because it was sufficient for almost all tweets.

## 4. Is the model case sensitive? If yes, how does it affect accuracy?

This model is **not case sensitive**.

Why:

- it is based on `bert-base-uncased`
- the tokenizer lowercases text
- the local run confirmed `tokenizer_do_lower_case = true`

### Effect on accuracy

Lowercasing has both benefits and drawbacks:

- it reduces vocabulary sparsity
- it usually helps robustness on noisy English text
- it removes information carried by capitalization

For sentiment analysis, losing case can slightly hurt performance when casing itself expresses emotion, for example:

- `I love this`
- `I LOVE THIS`
- `This is BAD`

An uncased model treats those more similarly than a cased model would. On social media text, this can remove useful sentiment emphasis.

Important note:

- I did **not** run a separate cased-vs-uncased ablation in this repo
- so the effect on accuracy here is a reasoned analysis, not a measured local comparison

## 5. Is it possible to use this model for agglutinative languages such as Azerbaijani?

**Not as a good production choice in its current form.**

Reasons:

- it was pretrained for English, not Azerbaijani
- it uses an English uncased vocabulary
- Azerbaijani is agglutinative, so many word forms are created by attaching suffixes
- English WordPiece segmentation is not optimized for Azerbaijani morphology
- lowercasing also removes casing distinctions that can matter in multilingual settings

So, while the tokenizer can still split unknown words into subwords, the model is not well matched to Azerbaijani sentiment analysis.

### Better alternatives for Azerbaijani

A stronger choice would be:

- `bert-base-multilingual-cased`
- XLM-R (`xlm-roberta-base` or `xlm-roberta-large`)
- a multilingual or Azerbaijani-adapted checkpoint fine-tuned on Azerbaijani sentiment data

Practical conclusion:

- this English SST-2 BERT model is suitable for English sentiment classification
- for Azerbaijani, a multilingual model should be selected and fine-tuned on Azerbaijani examples

## 6. Local test on `Sentiment140_v2.csv`

### What was tested

I ran the fine-tuned BERT model on a **balanced 2,000-row sample** from `Sentiment140_v2.csv`:

- 1,000 negative tweets
- 1,000 positive tweets
- random seed: `42`

Command used:

```bash
.venv\Scripts\python evaluate_sentiment140_bert.py --max-rows 2000 --batch-size 64 --max-length 64
```

### Dataset format used

The file contains:

- `polarity`
- `text`

Label mapping used in evaluation:

- `0 -> negative`
- `4 -> positive`

These were normalized internally to:

- `0 -> negative`
- `1 -> positive`

### Local results

Results from `outputs_task1_sentiment/metrics_textattack__bert-base-uncased-SST-2.json`:

- accuracy: **73.65%**
- macro-F1: **73.54%**
- positive precision: **77.09%**
- positive recall: **67.30%**
- negative precision: **70.98%**
- negative recall: **80.00%**

Confusion matrix:

- true negative: `800`
- false positive: `200`
- false negative: `327`
- true positive: `673`

### Interpretation

The model works reasonably well on short English tweets, but the score is clearly below what it would usually achieve on in-domain data such as SST-2.

That is expected because:

- the model was fine-tuned on SST-2, not Twitter
- tweets contain slang, usernames, emoji-style punctuation, and informal spelling
- Sentiment140 labels are tweet-style sentiment labels, not movie-review sentence labels

So the local test shows that the model is usable on `Sentiment140_v2.csv`, but it is not domain-matched.

## 7. Final conclusion

This Task 1 model is a valid open-source fine-tuned BERT sentiment classifier with:

- **input**: tokenized text (`input_ids`, `attention_mask`, optional `token_type_ids`)
- **output**: binary sentiment logits
- **number of classes**: `2`
- **maximum input size**: `512` tokens
- **case sensitivity**: uncased / not case sensitive
- **Azerbaijani suitability**: possible only in a weak fallback sense, but not a recommended choice

The local Sentiment140 experiment confirms that the model can be tested successfully in this repo and gives a reproducible baseline for the report.
