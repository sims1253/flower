# Tokenizers

The BPE tokenizer models (`fineweb_4k/8k/16k/32k.json`) are **not checked in**.
They are data artifacts that are cheap to regenerate, so they are gitignored and
must be (re)built before running any experiment. A missing tokenizer will fail
loudly when a config references it.

## Regenerate the fineweb tokenizers

Trained on a sample of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
(`sample-10BT` split) with [`scripts/train_custom_tokenizer.py`](../scripts/train_custom_tokenizer.py):

```bash
# Online (streams from Hugging Face):
uv run python scripts/train_custom_tokenizer.py --vocab-size 4096  --num-docs 20000 --output tokenizers/fineweb_4k.json
uv run python scripts/train_custom_tokenizer.py --vocab-size 8192  --num-docs 20000 --output tokenizers/fineweb_8k.json
uv run python scripts/train_custom_tokenizer.py --vocab-size 16384 --num-docs 20000 --output tokenizers/fineweb_16k.json
uv run python scripts/train_custom_tokenizer.py --vocab-size 32768 --num-docs 20000 --output tokenizers/fineweb_32k.json

# Offline, from pre-fetched parquet shards (faster for re-runs; see scripts/prefetch_dataset.py):
uv run python scripts/train_custom_tokenizer.py --vocab-size 32768 \
    --from-cache data_cache/sample/10BT --output tokenizers/fineweb_32k.json
```

The vocab size is a data-efficiency lever, not just a parameter-count knob: a
smaller vocab packs fewer bytes per token, so a fixed token budget sees less
text. The training script reports `bytes/token` on held-out docs to make that
trade explicit.

## Algorithm ablation set (`algo/`)

[`scripts/compare_tokenizer_algorithms.py`](../scripts/compare_tokenizer_algorithms.py)
trains a set of *different algorithms* (bpe_regex, bpe_noregex, unigram,
wordpiece) at matched vocab and writes them to `tokenizers/algo/`:

```bash
uv run python scripts/compare_tokenizer_algorithms.py --vocab 16384 \
    --cache data_cache/sample/10BT --docs 40000 --out tokenizers/algo
```

## Usage

Configs reference a tokenizer by path with the `custom:` prefix, e.g.:

```yaml
data:
  tokenizer: "custom:tokenizers/fineweb_4k.json"
```

Loaded by `flower.data.build_tokenizer("custom:<path>")`.
