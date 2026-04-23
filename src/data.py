"""
data.py — Dataset loading and tokenization for FiNER-ORD.

Handles:
  - Loading gtfintechlab/finer-ord from HuggingFace
  - Converting PER_B/PER_I suffix labels to B-PER/I-PER prefix (seqeval format)
  - Grouping token-flat data into sentence-level sequences
  - First-subword-only or all-subword label alignment for tokenization
  - Sanity checks for label mapping
"""

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


# The raw HF dataset ships integer labels (0-6). The paper names these
# O, PER_B, PER_I, LOC_B, LOC_I, ORG_B, ORG_I in that order. We map them
# to seqeval-style IOB2 prefix strings (B-PER, I-PER, ...) because that's
# the format seqeval's entity decoder expects.
ID2LABEL = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-LOC",
    4: "I-LOC",
    5: "B-ORG",
    6: "I-ORG",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)


def get_continuation_label_id(label_id):
    """Map a word-level BIO label to the label used on continuation subwords."""
    label_name = ID2LABEL[label_id]
    if label_name == "O" or label_name.startswith("I-"):
        return label_id
    if label_name.startswith("B-"):
        return LABEL2ID[f"I-{label_name[2:]}"]
    raise ValueError(f"Unsupported label for continuation subwords: {label_name}")


def align_labels_to_subwords(word_ids, word_labels, label_all_subwords=False):
    """Align one word-level label sequence to tokenizer subwords."""
    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(word_labels[word_idx])
        elif label_all_subwords:
            label_ids.append(get_continuation_label_id(word_labels[word_idx]))
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


def format_label_ids(label_ids):
    """Convert aligned integer labels into printable strings."""
    return ["-100" if label_id == -100 else ID2LABEL[label_id] for label_id in label_ids]


def load_finer_ord():
    """
    Load FiNER-ORD from HuggingFace and group token-flat rows into
    sentence-level sequences keyed by (doc_idx, sent_idx).

    Returns a DatasetDict with train/validation/test splits. Each example has:
      doc_idx, sent_idx, gold_token (list of str), gold_label (list of int).
    """
    raw = load_dataset("gtfintechlab/finer-ord")

    grouped_splits = {}
    for split_name, split_data in raw.items():
        df = split_data.to_pandas()
        df["gold_token"] = df["gold_token"].astype(str)

        grouped = (
            df.groupby(["doc_idx", "sent_idx"])
            .agg(gold_token=("gold_token", list), gold_label=("gold_label", list))
            .reset_index()
        )
        grouped_splits[split_name] = Dataset.from_pandas(grouped)

    return DatasetDict(grouped_splits)


def tokenize_dataset(dataset, tokenizer, max_length=256, label_all_subwords=False):
    """Apply tokenization and label alignment across all dataset splits."""

    def tokenize_and_align(examples):
        clean_tokens = [[str(tok) for tok in seq] for seq in examples["gold_token"]]
        tokenized = tokenizer(
            clean_tokens,
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
        )

        aligned_labels = []
        for i, word_labels in enumerate(examples["gold_label"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned_labels.append(
                align_labels_to_subwords(
                    word_ids,
                    word_labels,
                    label_all_subwords=label_all_subwords,
                )
            )

        tokenized["labels"] = aligned_labels
        return tokenized

    return dataset.map(tokenize_and_align, batched=True)


def sanity_check_labels(grouped_dataset, max_samples=3):
    """Print label mapping, example entity-bearing sentences, and split sizes."""
    print("=" * 60)
    print("Label mapping")
    print("=" * 60)
    for k, v in ID2LABEL.items():
        print(f"  {k} -> {v}")

    print("\nSentences containing entities:")
    found = 0
    for i, row in enumerate(grouped_dataset["train"]):
        if not any(lbl != 0 for lbl in row["gold_label"]):
            continue
        print(f"\n--- Sentence index {i} ---")
        for tok, lbl_id in zip(row["gold_token"], row["gold_label"]):
            print(f"  {tok:25s} {ID2LABEL[lbl_id]}")
        found += 1
        if found >= max_samples:
            break

    print("\n" + "=" * 60)
    print("Split statistics")
    print("=" * 60)
    for split_name, split_data in grouped_dataset.items():
        n_sentences = len(split_data)
        n_tokens = sum(len(x["gold_token"]) for x in split_data)
        print(f"  {split_name}: {n_sentences} sentences, {n_tokens} tokens")


def verify_alignment(
    grouped_dataset,
    tokenizer,
    max_length=256,
    sample_idx=0,
    label_all_subwords=False,
):
    """Print subword-level alignment for one example to eyeball tokenizer behavior."""
    print("=" * 60)
    print(f"Subword alignment for train[{sample_idx}]")
    print("=" * 60)

    row = grouped_dataset["train"][sample_idx]
    clean_tokens = [str(t) for t in row["gold_token"]]
    tokenized = tokenizer(
        [clean_tokens],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )
    word_ids = tokenized.word_ids(batch_index=0)
    subword_tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
    aligned_labels = align_labels_to_subwords(
        word_ids,
        row["gold_label"],
        label_all_subwords=label_all_subwords,
    )

    previous_word_idx = None
    for subword, word_idx, label_id in zip(subword_tokens, word_ids, aligned_labels):
        if label_id == -100 and word_idx is None:
            label = "-100 (special)"
        elif label_id == -100 and word_idx == previous_word_idx:
            label = "-100 (continuation)"
        else:
            label = ID2LABEL[label_id]
        print(f"  {subword:20s} {label}")
        previous_word_idx = word_idx


def run_label_alignment_demo(tokenizer, max_length=256):
    """Print a small ORG example under both alignment modes."""
    print("\n" + "=" * 60)
    print("Label alignment demo: Apple Inc.")
    print("=" * 60)

    example_tokens = ["Apple", "Inc."]
    example_labels = [LABEL2ID["B-ORG"], LABEL2ID["I-ORG"]]

    tokenized = tokenizer(
        [example_tokens],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )
    word_ids = tokenized.word_ids(batch_index=0)
    subword_tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])

    first_subword_only = align_labels_to_subwords(
        word_ids,
        example_labels,
        label_all_subwords=False,
    )
    all_subwords = align_labels_to_subwords(
        word_ids,
        example_labels,
        label_all_subwords=True,
    )

    continuation_positions = []
    seen_word_indices = set()
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx in seen_word_indices:
            continuation_positions.append(idx)
        else:
            seen_word_indices.add(word_idx)

    special_positions = [idx for idx, word_idx in enumerate(word_ids) if word_idx is None]
    inc_continuation_positions = [idx for idx in continuation_positions if word_ids[idx] == 1]

    assert inc_continuation_positions, "Expected 'Inc.' to split into continuation subwords."
    assert all(first_subword_only[idx] == -100 for idx in continuation_positions)
    assert all(all_subwords[idx] == LABEL2ID["I-ORG"] for idx in inc_continuation_positions)
    assert all(first_subword_only[idx] == -100 for idx in special_positions)
    assert all(all_subwords[idx] == -100 for idx in special_positions)

    print(f"  Words:          {example_tokens}")
    print(f"  Subwords:       {subword_tokens}")
    print(f"  Word IDs:       {word_ids}")
    print(f"  Default labels: {format_label_ids(first_subword_only)}")
    print(f"  All-subwords:   {format_label_ids(all_subwords)}")


def get_dataset_and_tokenizer(
    model_name,
    max_length=256,
    run_checks=False,
    label_all_subwords=False,
):
    """
    One-shot loader for training scripts.

    Returns (tokenized_dataset, tokenizer, grouped_dataset). The grouped
    dataset is kept around for error analysis on raw tokens/labels.
    """

    # RoBERTa-family tokenizers need add_prefix_space for split tokens
    if "roberta" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    grouped = load_finer_ord()

    if run_checks:
        sanity_check_labels(grouped)
        verify_alignment(
            grouped,
            tokenizer,
            max_length=max_length,
            label_all_subwords=label_all_subwords,
        )

    tokenized = tokenize_dataset(
        grouped,
        tokenizer,
        max_length=max_length,
        label_all_subwords=label_all_subwords,
    )
    return tokenized, tokenizer, grouped


if __name__ == "__main__":
    # Smoke test: python -m src.data
    _, tokenizer, grouped = get_dataset_and_tokenizer("roberta-base", run_checks=False)
    sanity_check_labels(grouped)
    verify_alignment(grouped, tokenizer)
    run_label_alignment_demo(tokenizer)
