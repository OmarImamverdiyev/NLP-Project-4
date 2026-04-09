# Task 1 Extended: Azerbaijani Sentiment Model

## Does the current Task 1 already satisfy the assignment?

Yes.

The original assignment says to analyze **any open-source fine-tuned BERT model for sentiment analysis** and answer:

- inputs and outputs
- number of classes
- input size
- case sensitivity
- whether it is suitable for Azerbaijani

That is already covered in:

- `task1_sentiment_analysis.md`

using the open-source model:

- `textattack/bert-base-uncased-SST-2`

So the repository already satisfies Task 1 at the assignment level.

## Why this extension exists

The original Task 1 model is English-only, so the Azerbaijani discussion in the report is mostly analytical.

This `task1_extended/` folder adds a practical Azerbaijani-focused extension using the Hugging Face model:

- `StartZer0/az-sentiment-bert`

Model page:

- https://huggingface.co/StartZer0/az-sentiment-bert

According to the model card, it is:

- language: Azerbaijani (`az`)
- task: sentiment analysis
- base model: `allmalab/bert-base-aze`
- dataset: `hajili/azerbaijani_review_sentiment_classification`

## What was verified locally

I verified locally that this model can be loaded directly from the internet with `transformers`:

- tokenizer type: `BertTokenizer`
- `num_labels = 2`
- `model_max_length = 512`
- `do_lower_case = True`

I also tested a few Azerbaijani sentences and confirmed the prediction direction is sensible:

- positive-style sentences map to class `1`
- negative-style sentences map to class `0`

This means the practical sentiment mapping is:

- `0 -> negative`
- `1 -> positive`

## Files in this folder

- `predict_az_sentiment.py`: downloads the model from Hugging Face on first use and runs sentiment prediction
- `download_model.ps1`: optional Windows helper that installs `git-xet`, installs the Hugging Face CLI, and downloads the model locally

## Run directly from the internet

This is the simplest option. The model is downloaded automatically by `transformers` and cached locally.

```bash
.venv\Scripts\python task1_extended\predict_az_sentiment.py --text "Bu mehsul cox yaxsidir." --text "Bu xidmet berbad idi."
```

You can also put one text per line in a file:

```bash
.venv\Scripts\python task1_extended\predict_az_sentiment.py --input-file task1_extended\sample_texts.txt
```

## Optional: download the model explicitly

If you want to download the model yourself first, you can use the helper script:

```powershell
powershell -ExecutionPolicy Bypass -File task1_extended\download_model.ps1
```

Or run the commands manually:

```powershell
winget install git-xet
git clone https://huggingface.co/StartZer0/az-sentiment-bert
```

If you want just pointer files:

```powershell
$env:GIT_LFS_SKIP_SMUDGE="1"
git clone https://huggingface.co/StartZer0/az-sentiment-bert
```

And with the Hugging Face CLI:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
hf download StartZer0/az-sentiment-bert
```

## Notes

- You do **not** need `git-xet` just to run `predict_az_sentiment.py`
- `git-xet` is only useful if you want to clone/download model files manually
- the model is a much better Azerbaijani fit than the English SST-2 BERT model used in the original Task 1 report
