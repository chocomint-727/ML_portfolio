"""
ArXiv MLM Pretraining Pipeline (Chunked Version)
================================================

End-to-end:
1. Download scientific_papers/arxiv
2. Clean text
3. Split train/val
4. Save clean dataset
5. Build tokenizer
6. Chunk + tokenize (sliding window MLM)
7. Save tokenized dataset
8. Generate reusable dataloader module

Output:
    arxiv_data/
"""

import os
import re
import logging
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_DIR      = "arxiv_data"
RAW_DIR       = os.path.join(BASE_DIR, "raw_clean")
TOKEN_DIR     = os.path.join(BASE_DIR, "tokenized_chunked")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")

TOKENIZER_NAME = "allenai/scibert_scivocab_uncased"

MAX_LENGTH = 512
STRIDE     = 128
VAL_SIZE    = 20000
NUM_PROC    = 2

os.makedirs(BASE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. DOWNLOAD
# ─────────────────────────────────────────────

def download():
    log.info("Downloading scientific_papers/arxiv...")

    ds = load_dataset(
        "scientific_papers",
        "arxiv",
        split="train",
        trust_remote_code=True,
    )

    log.info(f"Loaded {len(ds):,} samples")
    return ds

# ─────────────────────────────────────────────
# 2. CLEANING
# ─────────────────────────────────────────────

def clean_text(text):
    text = text or ""

    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def process(example):
    return {
        "text": clean_text(example.get("article", "")),
        "abstract": clean_text(example.get("abstract", "")),
    }


def is_valid(example):
    return len(example["text"]) > 500

# ─────────────────────────────────────────────
# 3. CLEAN + SPLIT
# ─────────────────────────────────────────────

def build_clean_dataset(raw):
    log.info("Cleaning dataset...")

    ds = raw.map(
        process,
        remove_columns=raw.column_names,
        num_proc=NUM_PROC,
    )

    ds = ds.filter(
        is_valid,
        num_proc=NUM_PROC,
    )

    split = ds.train_test_split(test_size=VAL_SIZE, seed=42)

    clean_ds = DatasetDict({
        "train": split["train"],
        "val": split["test"],
    })

    clean_ds.save_to_disk(RAW_DIR)

    log.info(f"Saved clean dataset to {RAW_DIR}")

    return clean_ds

# ─────────────────────────────────────────────
# 4. TOKENIZER
# ─────────────────────────────────────────────

def build_tokenizer():
    log.info("Building tokenizer...")

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tok.save_pretrained(TOKENIZER_DIR)

    return tok

# ─────────────────────────────────────────────
# 5. CHUNKED TOKENIZATION (IMPORTANT PART)
# ─────────────────────────────────────────────

def tokenize_and_chunk(dataset, tokenizer):
    log.info("Chunking + tokenizing dataset...")

    def fn(batch):
        texts = [
            (a + " " + tokenizer.sep_token + " " + t).strip()
            if a else t
            for a, t in zip(batch["abstract"], batch["text"])
        ]

        encoded = tokenizer(
            texts,
            truncation=False,
            padding=False,
        )

        input_ids_all = []
        attention_all = []

        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):

            start = 0

            while start < len(ids):
                end = start + MAX_LENGTH

                chunk_ids = ids[start:end]
                chunk_mask = mask[start:end]

                pad_len = MAX_LENGTH - len(chunk_ids)

                if pad_len > 0:
                    chunk_ids += [tokenizer.pad_token_id] * pad_len
                    chunk_mask += [0] * pad_len

                input_ids_all.append(chunk_ids)
                attention_all.append(chunk_mask)

                start += (MAX_LENGTH - STRIDE)

        return {
            "input_ids": input_ids_all,
            "attention_mask": attention_all,
        }

    tokenized = dataset.map(
        fn,
        batched=True,
        batch_size=32,
        num_proc=NUM_PROC,
        remove_columns=dataset["train"].column_names,
    )

    tokenized.save_to_disk(TOKEN_DIR)

    log.info(f"Saved tokenized dataset to {TOKEN_DIR}")

    return tokenized

# ─────────────────────────────────────────────
# 6. DATALOADER MODULE GENERATION
# ─────────────────────────────────────────────

def write_dataloader():
    code = f'''
from datasets import load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

DATA_DIR = "{TOKEN_DIR}"
TOKENIZER_DIR = "{TOKENIZER_DIR}"

def load_data(batch_size=32, num_workers=2):
    ds = load_from_disk(DATA_DIR)

    train_loader = DataLoader(
        ds["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        ds["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR)
'''

    path = os.path.join(BASE_DIR, "dataloader.py")

    with open(path, "w") as f:
        f.write(code)

    log.info(f"Saved dataloader module to {path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    raw = download()

    clean = build_clean_dataset(raw)

    tokenizer = build_tokenizer()

    tokenize_and_chunk(clean, tokenizer)

    write_dataloader()

    log.info("PIPELINE COMPLETE 🚀")