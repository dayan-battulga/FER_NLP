from __future__ import annotations

import argparse


def inspect_hf_dataset(
    name: str = "gtfintechlab/finer-ord",
    n_rows: int = 5,
    string_sample: int = 1000,
) -> None:
    """Load a HuggingFace dataset and print splits, features, column
    dtypes, ClassLabel names, integer/string stats, a handful of raw
    rows, and (if token-flat) one reconstructed entity-bearing sentence.
    """
    from datasets import ClassLabel, Sequence, Value, load_dataset

    def describe(feat) -> str:
        if isinstance(feat, Value):
            return f"Value({feat.dtype})"
        if isinstance(feat, ClassLabel):
            return f"ClassLabel(n={feat.num_classes})"
        if isinstance(feat, Sequence):
            return f"Sequence[{describe(feat.feature)}]"
        return repr(feat)

    print(f"Loading {name} ...")
    ds = load_dataset(name)

    # 1. Splits.
    print("\n=== Splits ===")
    for split, d in ds.items():
        print(f"  {split:12s}  {len(d):>7d} rows   cols: {d.column_names}")

    # 2. Features per split.
    print("\n=== Features (per split) ===")
    for split, d in ds.items():
        print(f"  [{split}]")
        for col, feat in d.features.items():
            print(f"    {col:18s}  {describe(feat)}")

    # 3. Column-by-column details on train (or first split).
    train = ds["train"] if "train" in ds else ds[next(iter(ds))]
    print(f"\n=== Column details ({'train' if 'train' in ds else next(iter(ds))}) ===")
    for col, feat in train.features.items():
        print(f"\n  {col}:")
        print(f"    feature: {feat}")

        if isinstance(feat, ClassLabel):
            for i, n in enumerate(feat.names):
                print(f"      class {i} -> {n}")

        vals = train[col]
        if isinstance(feat, Value):
            if str(feat.dtype).startswith(("int", "uint")):
                try:
                    print(
                        f"    min={min(vals)}  max={max(vals)}  "
                        f"n_unique={len(set(vals))}"
                    )
                except Exception as e:
                    print(f"    (int stats skipped: {e})")
            elif feat.dtype == "string":
                head = vals[: min(len(vals), string_sample)]
                lens = [len(x) for x in head]
                print(
                    f"    string len over first {len(head)}: "
                    f"min={min(lens)} mean={sum(lens) / len(lens):.1f} "
                    f"max={max(lens)}"
                )

        preview = vals[: min(5, len(vals))]
        print(f"    first 5 values: {preview}")

    # 4. Raw row dump.
    print(
        f"\n=== First {n_rows} raw rows ({'train' if 'train' in ds else next(iter(ds))}) ==="
    )
    for i in range(min(n_rows, len(train))):
        print(f"  row {i}: {train[i]}")

    # 5. If token-flat, reconstruct one entity-bearing sentence so the
    # integer label IDs are visible next to their tokens.
    is_flat = "doc_idx" in train.column_names and not any(
        isinstance(f, Sequence) for f in train.features.values()
    )
    if is_flat:
        print("\n=== Token-flat data detected; first entity-bearing sentence ===")
        token_col = next(
            (
                c
                for c in ("gold_token", "goldtoken", "token")
                if c in train.column_names
            ),
            None,
        )
        label_col = next(
            (
                c
                for c in ("gold_label", "goldlabel", "label")
                if c in train.column_names
            ),
            None,
        )
        group_cols = [c for c in ("doc_idx", "sent_idx") if c in train.column_names]
        if token_col and label_col and group_cols:
            df = train.to_pandas()
            found = False
            for key, group in df.groupby(group_cols):
                labels = group[label_col].tolist()
                if any(int(x) != 0 for x in labels):
                    print(f"    group key {group_cols} = {key}")
                    for t, l in zip(group[token_col].tolist(), labels):
                        marker = "" if int(l) == 0 else f"   <-- label_id {l}"
                        print(f"    {str(t):25s} {marker}")
                    found = True
                    break
            if not found:
                print("    (no entity-bearing sentence found in train)")
        else:
            missing = []
            if not token_col:
                missing.append("token column")
            if not label_col:
                missing.append("label column")
            if not group_cols:
                missing.append("grouping column(s)")
            print(f"    could not reconstruct: missing {missing}")
    else:
        print(
            "\n=== Data appears sentence-grouped (has Sequence features); "
            "no reconstruction needed ==="
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a HuggingFace dataset.")
    ap.add_argument("--name", default="gtfintechlab/finer-ord")
    ap.add_argument("--n-rows", type=int, default=5)
    args = ap.parse_args()
    inspect_hf_dataset(name=args.name, n_rows=args.n_rows)


if __name__ == "__main__":
    main()
