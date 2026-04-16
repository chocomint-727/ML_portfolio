"""
PubMed MLM Data Pipeline
========================
Downloads, filters, tokenizes, and saves the ncbi/pubmed dataset
ready to be plugged into the MLMTrainer architecture.
 
Usage:
    python pubmed_pipeline.py
 
Output:
    ./pubmed_tokenized/   — Arrow cache, load with datasets.load_from_disk()
"""
 
import os
import logging
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────
# Config — edit these
# ─────────────────────────────────────────────
 
TOKENIZER_NAME   = "allenai/scibert_scivocab_uncased"  # biomedical vocab; swap for bert-base-uncased if preferred
MAX_LENGTH       = 512
MIN_ABSTRACT_LEN = 100    # characters — filters near-empty abstracts
VAL_SIZE         = 50_000 # held-out examples for validation loss tracking
BATCH_SIZE       = 32
NUM_PROC         = os.cpu_count()   # parallel workers for .map()
OUTPUT_DIR       = "./pubmed_tokenized"
CACHE_DIR        = None   # set to e.g. "/scratch/.hf_cache" if home disk is small
 
# ─────────────────────────────────────────────
# 1. Download
# ─────────────────────────────────────────────
 
def download_dataset(cache_dir: str | None) -> "Dataset":
    log.info("Downloading ncbi/pubmed (~20 GB, may take a while)…")
    ds = load_dataset(
        "ncbi/pubmed",
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    # dataset has no predefined splits — it's one flat 'train' key
    raw = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    log.info(f"Downloaded {len(raw):,} raw records")
    return raw
 
# ─────────────────────────────────────────────
# 2. Extract & filter
# ─────────────────────────────────────────────
 
def extract_text(example: dict) -> dict:
    """Pull abstract + title out of the deeply nested XML-derived schema."""
    try:
        article  = example["MedlineCitation"]["Article"]
        abstract = article["Abstract"]["AbstractText"]
        title    = article.get("ArticleTitle", "")
        language = article.get("Language", "")
    except (KeyError, TypeError):
        return {"text": "", "language": ""}
    return {"text": abstract, "title": title, "language": language}
 
 
def is_usable(example: dict) -> bool:
    """Keep only substantive English abstracts."""
    return (
        example["language"] == "en"
        and len(example["text"]) >= MIN_ABSTRACT_LEN
    )
 
 
def filter_and_extract(raw) -> "Dataset":
    log.info("Extracting text fields…")
    extracted = raw.map(
        extract_text,
        num_proc=NUM_PROC,
        remove_columns=raw.column_names,
        desc="Extracting",
    )
    log.info("Filtering (English, non-empty abstracts)…")
    filtered = extracted.filter(
        is_usable,
        num_proc=NUM_PROC,
        desc="Filtering",
    )
    log.info(f"Kept {len(filtered):,} records after filtering")
    return filtered
 
# ─────────────────────────────────────────────
# 3. Tokenize
# ─────────────────────────────────────────────
 
def build_tokenizer(name: str) -> AutoTokenizer:
    log.info(f"Loading tokenizer: {name}")
    tok = AutoTokenizer.from_pretrained(name)
    log.info(
        f"Vocab size: {tok.vocab_size:,} | "
        f"[MASK] id: {tok.mask_token_id} | "
        f"[PAD] id: {tok.pad_token_id}"
    )
    return tok
 
 
def make_tokenize_fn(tokenizer, max_length: int):
    def tokenize(batch: dict) -> dict:
        # Concatenate title + abstract so the model sees full context.
        # A separator token naturally breaks them.
        texts = [
            (t + " " + tokenizer.sep_token + " " + a).strip()
            if t else a
            for t, a in zip(batch["title"], batch["text"])
        ]
        return tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,  # not needed for encoder-only MLM
        )
    return tokenize
 
 
def tokenize_dataset(filtered, tokenizer, max_length: int) -> "Dataset":
    log.info("Tokenizing…")
    tokenize_fn = make_tokenize_fn(tokenizer, max_length)
    tokenized = filtered.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROC,
        remove_columns=filtered.column_names,
        desc="Tokenizing",
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized
 
# ─────────────────────────────────────────────
# 4. Train / val split & save
# ─────────────────────────────────────────────
 
def split_and_save(tokenized, val_size: int, output_dir: str) -> DatasetDict:
    log.info(f"Splitting off {val_size:,} examples for validation…")
    split = tokenized.train_test_split(test_size=val_size, seed=42)
    ds = DatasetDict({"train": split["train"], "val": split["test"]})
 
    log.info(f"Saving to {output_dir}…")
    ds.save_to_disk(output_dir)
    log.info(
        f"Saved — train: {len(ds['train']):,} | val: {len(ds['val']):,}"
    )
    return ds
 
# ─────────────────────────────────────────────
# 5. DataLoader factory (import this in your training script)
# ─────────────────────────────────────────────
 
def get_dataloaders(
    data_dir: str = OUTPUT_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Load the pre-processed dataset from disk and return train/val DataLoaders.
    Call this in your training script after running the pipeline once.
 
    Example:
        train_loader, val_loader = get_dataloaders()
        for batch in train_loader:
            input_ids     = batch["input_ids"].to(device)      # (B, 512)
            padding_mask  = (input_ids == pad_id)              # (B, 512) bool
            logits, ground = mlm_trainer(input_ids, padding_mask)
            loss = loss_fn(logits.view(-1, vocab_size), ground.view(-1))
    """
    ds = load_from_disk(data_dir)
    train_loader = DataLoader(
        ds["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,   # keeps batch size uniform — avoids edge cases in LayerNorm
    )
    val_loader = DataLoader(
        ds["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
 
# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR):
        log.info(f"Found existing processed dataset at {OUTPUT_DIR}, skipping pipeline.")
        log.info("Delete the directory to re-run from scratch.")
    else:
        tokenizer = build_tokenizer(TOKENIZER_NAME)
 
        raw      = download_dataset(CACHE_DIR)
        filtered = filter_and_extract(raw)
        tokenized = tokenize_dataset(filtered, tokenizer, MAX_LENGTH)
        split_and_save(tokenized, VAL_SIZE, OUTPUT_DIR)
 
    # Quick sanity check
    log.info("Running sanity check on saved dataset…")
    train_loader, val_loader = get_dataloaders(batch_size=4, num_workers=0)
    batch = next(iter(train_loader))
    log.info(f"input_ids shape:     {batch['input_ids'].shape}")       # (4, 512)
    log.info(f"attention_mask shape: {batch['attention_mask'].shape}") # (4, 512)
    log.info("Pipeline complete. Import get_dataloaders() in your training script.")