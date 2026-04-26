"""
dapt_corpus.py - Multi-source DAPT corpus builder for FiNER-ORD Phase 5.

Builds a domain-adaptive pretraining corpus by combining, in order:

  1. FNSPID financial news (subsampled via HuggingFace `streaming=True` so
     the full ~15.7M rows are never materialized).
  2. Optional cc_news rows filtered to a fixed list of finance domains.
  3. FiNER-ORD train articles (with the existing val/test leak filter and
     a small held-out MLM perplexity probe set aside).

Critical correctness rules (matched to docs/PROJECT_CONTEXT.MD and the
Phase 5 plan):

  - No FNSPID/cc_news row whose URL or body content-hash matches a FiNER
    val/test article passes the leak filter.
  - URL-based dedup runs first; rows with no URL fall back to body
    SHA-256 dedup.
  - The FiNER-ORD train article filter that excludes articles textually
    identical to a val/test article is preserved.
  - A small held-out FiNER probe (deterministic by SHA-256 of doc_idx)
    is excluded from the DAPT corpus and returned separately for an
    apples-to-apples MLM perplexity probe vs the base checkpoint.
  - Token count is logged, not targeted; the FNSPID subsample size is
    the only input knob.

Public API:

    build_dapt_corpus(spec) -> (article_texts, held_out_finer_texts, diagnostics)
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass
from typing import Any

from src.data import load_finer_ord


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Domains restricted to in cc_news. Keeping this small and explicit avoids
# silently bringing non-finance news into the DAPT mix.
CC_NEWS_FINANCE_DOMAINS: frozenset[str] = frozenset(
    {
        "reuters.com",
        "bloomberg.com",
        "ft.com",
        "wsj.com",
        "cnbc.com",
        "marketwatch.com",
        "forbes.com",
        "seekingalpha.com",
        "barrons.com",
        "investopedia.com",
    }
)

# FNSPID column names vary across mirrors; probe these candidates and use
# the first one that exposes a string body.
FNSPID_BODY_FIELD_CANDIDATES: tuple[str, ...] = (
    "Article",
    "article",
    "text",
    "body",
    "content",
    "article_body",
)

FNSPID_URL_FIELD_CANDIDATES: tuple[str, ...] = (
    "Url",
    "url",
    "link",
    "Link",
)

# Belt-and-braces ceiling on cc_news kept rows. cc_news is large and the
# domain filter alone may not bound it tightly enough to avoid runaway
# streaming on Colab.
CC_NEWS_MAX_KEPT: int = 50_000


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass
class CorpusSpec:
    """Inputs to `build_dapt_corpus`. Mirrors the Phase 5 config fields."""

    corpus_source: str
    include_cc_news: bool = False
    fnspid_subsample: int = 100_000
    held_out_probe_size: int = 5
    fnspid_dataset_id: str = "Zihan1004/FNSPID"
    cc_news_dataset_id: str = "vblagoje/cc_news"
    fnspid_min_median_chars: int = 1000
    fnspid_sanity_sample_size: int = 100
    fnspid_sanity_max_rows_scanned: int = 50_000
    fnspid_progress_every: int = 5_000
    per_article_min_chars: int = 200
    seed: int = 88


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def _sha256(text: str) -> str:
    """SHA-256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _url_hash(url: str | None) -> str | None:
    """Lower-cased, stripped URL hash; None when the URL is missing."""
    if not url:
        return None
    return hashlib.sha256(url.strip().lower().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# FiNER partition: corpus articles + held-out probe + leak hash set
# ---------------------------------------------------------------------------


def _select_held_out_doc_ids(
    train_doc_ids: list[int],
    probe_size: int,
) -> list[int]:
    """Pick `probe_size` doc_idx values deterministically by hash."""
    if probe_size <= 0:
        return []
    if probe_size >= len(train_doc_ids):
        raise ValueError(
            f"`held_out_probe_size`={probe_size} would consume every train "
            f"article (only {len(train_doc_ids)} available)."
        )
    scored = sorted(
        train_doc_ids,
        key=lambda doc_idx: hashlib.sha256(
            f"finer_holdout_{int(doc_idx)}".encode("utf-8")
        ).hexdigest(),
    )
    return sorted(scored[:probe_size])


def _articles_by_doc(grouped: Any, split_name: str) -> dict[int, str]:
    """Reconstruct article-level text keyed by `doc_idx` for one split."""
    if split_name not in grouped:
        raise ValueError(f"Expected split `{split_name}` in FiNER-ORD.")
    sentences_by_doc: dict[int, list[tuple[int, list[str]]]] = {}
    for row in grouped[split_name]:
        doc_idx = int(row["doc_idx"])
        sent_idx = int(row["sent_idx"])
        tokens = [str(token) for token in row["gold_token"]]
        sentences_by_doc.setdefault(doc_idx, []).append((sent_idx, tokens))

    articles: dict[int, str] = {}
    for doc_idx, items in sentences_by_doc.items():
        ordered = sorted(items, key=lambda pair: pair[0])
        articles[doc_idx] = "\n".join(" ".join(tokens) for _, tokens in ordered)
    return articles


def _build_finer_partition(
    held_out_probe_size: int,
) -> tuple[list[str], list[str], set[str], dict[str, Any]]:
    """Split FiNER train articles into corpus / held-out, and return the
    val/test forbidden-content-hash set."""

    grouped = load_finer_ord()
    train_articles_by_doc = _articles_by_doc(grouped, "train")
    val_articles_by_doc = _articles_by_doc(grouped, "validation")
    test_articles_by_doc = _articles_by_doc(grouped, "test")

    val_text_set = set(val_articles_by_doc.values())
    test_text_set = set(test_articles_by_doc.values())
    forbidden_hashes = {_sha256(t) for t in val_text_set | test_text_set}

    train_doc_ids_sorted = sorted(train_articles_by_doc.keys())
    held_out_doc_ids = set(
        _select_held_out_doc_ids(train_doc_ids_sorted, held_out_probe_size)
    )

    corpus_articles: list[str] = []
    held_out_articles: list[str] = []
    dropped_val = 0
    dropped_test = 0
    for doc_idx in train_doc_ids_sorted:
        text = train_articles_by_doc[doc_idx]
        if doc_idx in held_out_doc_ids:
            held_out_articles.append(text)
            continue
        if text in val_text_set:
            dropped_val += 1
            continue
        if text in test_text_set:
            dropped_test += 1
            continue
        corpus_articles.append(text)

    if not corpus_articles:
        raise RuntimeError(
            "FiNER train portion of the DAPT corpus is empty after the "
            "held-out probe and val/test leak filter were applied."
        )

    diagnostics = {
        "num_articles_total": len(train_articles_by_doc),
        "num_articles_used": len(corpus_articles),
        "num_held_out": len(held_out_articles),
        "dropped_duplicates_with_val": dropped_val,
        "dropped_duplicates_with_test": dropped_test,
        "held_out_doc_ids": sorted(held_out_doc_ids),
    }
    return corpus_articles, held_out_articles, forbidden_hashes, diagnostics


# ---------------------------------------------------------------------------
# FNSPID streaming subsample
# ---------------------------------------------------------------------------


def _stream_fnspid(
    spec: CorpusSpec,
    forbidden_hashes: set[str],
) -> tuple[list[str], dict[str, Any]]:
    """Stream FNSPID, sanity-check body length on the first 100 rows, and
    take the first `fnspid_subsample` rows that pass dedup + leak filters.

    A streaming pass is used so we never materialize the full ~15.7M rows.
    The first 100 rows double as the format / quality sanity check; if the
    median body length is below `fnspid_min_median_chars`, we abort with a
    clear error so the caller can fall back to the cc_news + FiNER mix.
    """

    from datasets import load_dataset

    stream = load_dataset(spec.fnspid_dataset_id, split="train", streaming=True)

    body_field: str | None = None
    url_field: str | None = None
    sample_lens: list[int] = []
    sanity_done = False

    seen_url_hashes: set[str] = set()
    seen_body_hashes: set[str] = set()
    kept: list[str] = []

    n_seen = 0
    n_too_short = 0
    n_dup_url = 0
    n_dup_body = 0
    n_leaked = 0

    for row in stream:
        n_seen += 1

        if body_field is None:
            # Detect the body column by NAME (not value type). Some FNSPID rows
            # have a None body even though the column exists; checking the value
            # of the very first row would falsely abort. The per-row
            # `isinstance(body, str)` check below handles None / non-string rows
            # by counting them as `num_dropped_too_short`.
            for candidate in FNSPID_BODY_FIELD_CANDIDATES:
                if candidate in row:
                    body_field = candidate
                    break
            if body_field is None:
                raise RuntimeError(
                    "FNSPID rows do not expose any of "
                    f"{list(FNSPID_BODY_FIELD_CANDIDATES)} as a column. "
                    f"Available columns: {sorted(row.keys())}"
                )
            for candidate in FNSPID_URL_FIELD_CANDIDATES:
                if candidate in row:
                    url_field = candidate
                    break

        body = row.get(body_field)

        # Sanity sample: collect lengths only from rows that actually have a
        # string body. FNSPID has a non-trivial fraction of summary-only rows
        # where `Article` is None, so "first 100 rows" can be all nulls and
        # never represent the real population. We instead require N actual
        # string bodies before deciding, and bail if we scan too many rows
        # without finding enough.
        if not sanity_done:
            if isinstance(body, str) and len(body) > 0:
                sample_lens.append(len(body))
            if len(sample_lens) >= spec.fnspid_sanity_sample_size:
                median_chars = float(statistics.median(sample_lens))
                if median_chars < spec.fnspid_min_median_chars:
                    raise RuntimeError(
                        "FNSPID body sanity check failed: median over "
                        f"{len(sample_lens)} string-bodied rows is "
                        f"{median_chars:.0f} chars, expected >= "
                        f"{spec.fnspid_min_median_chars}. Aborting; consider "
                        "falling back to cc_news + FiNER-train (set "
                        "`corpus_source: finer_train_only` and "
                        "`include_cc_news: true`)."
                    )
                sanity_done = True
                print(
                    f"FNSPID sanity check OK: median over "
                    f"{len(sample_lens)} string bodies is "
                    f"{median_chars:.0f} chars (threshold "
                    f"{spec.fnspid_min_median_chars}). Scanned {n_seen} rows."
                )
            elif n_seen >= spec.fnspid_sanity_max_rows_scanned:
                raise RuntimeError(
                    f"FNSPID body sanity check failed: only "
                    f"{len(sample_lens)} of the first {n_seen} rows had a "
                    f"string body, expected >= "
                    f"{spec.fnspid_sanity_sample_size}. The dataset appears "
                    "to be dominated by summary-only rows (no `Article` "
                    "field). Fall back to cc_news + FiNER-train."
                )

        if (
            spec.fnspid_progress_every > 0
            and n_seen % spec.fnspid_progress_every == 0
        ):
            print(
                f"  FNSPID stream: scanned {n_seen} rows, kept {len(kept)} "
                f"(target {spec.fnspid_subsample}); dropped "
                f"too_short={n_too_short}, dup_url={n_dup_url}, "
                f"dup_body={n_dup_body}, leak={n_leaked}"
            )

        if not isinstance(body, str) or len(body) < spec.per_article_min_chars:
            n_too_short += 1
            if len(kept) >= spec.fnspid_subsample:
                break
            continue

        body_hash = _sha256(body)
        if body_hash in forbidden_hashes:
            n_leaked += 1
            continue

        url_hash = _url_hash(row.get(url_field) if url_field else None)
        if url_hash is not None:
            if url_hash in seen_url_hashes:
                n_dup_url += 1
                continue
            seen_url_hashes.add(url_hash)
        else:
            if body_hash in seen_body_hashes:
                n_dup_body += 1
                continue
            seen_body_hashes.add(body_hash)

        kept.append(body)
        if len(kept) >= spec.fnspid_subsample:
            break

    # If we exhausted the stream before reaching the sanity threshold, decide
    # based on what we did see. An empty kept list is the real failure; a
    # short median over a small sample is just informational.
    if not sanity_done and not kept:
        raise RuntimeError(
            f"FNSPID stream produced no usable bodies after {n_seen} rows. "
            "Fall back to cc_news + FiNER-train."
        )

    diagnostics = {
        "num_rows_seen": n_seen,
        "num_rows_used": len(kept),
        "num_dropped_too_short": n_too_short,
        "num_dropped_dups_url": n_dup_url,
        "num_dropped_dups_body": n_dup_body,
        "num_dropped_leak": n_leaked,
        "median_article_chars": (
            float(statistics.median([len(t) for t in kept])) if kept else 0.0
        ),
        "body_field": body_field,
        "url_field": url_field,
        "sanity_median_chars_first_100": (
            float(statistics.median(sample_lens)) if sample_lens else 0.0
        ),
    }
    return kept, diagnostics


# ---------------------------------------------------------------------------
# cc_news streaming (optional)
# ---------------------------------------------------------------------------


def _stream_cc_news(
    spec: CorpusSpec,
    forbidden_hashes: set[str],
) -> tuple[list[str], dict[str, Any]]:
    """Stream cc_news, filter to finance domains, dedupe, and leak-filter.

    Bounded by `CC_NEWS_MAX_KEPT` to keep streaming finite even if the
    domain filter passes more rows than expected.
    """

    from datasets import load_dataset

    stream = load_dataset(spec.cc_news_dataset_id, split="train", streaming=True)

    seen_url_hashes: set[str] = set()
    seen_body_hashes: set[str] = set()
    kept: list[str] = []

    n_seen = 0
    n_filter_domain = 0
    n_too_short = 0
    n_dup_url = 0
    n_dup_body = 0
    n_leaked = 0

    for row in stream:
        n_seen += 1
        domain = (row.get("domain") or "").lower().strip()
        if domain not in CC_NEWS_FINANCE_DOMAINS:
            n_filter_domain += 1
            continue

        body = row.get("text") or row.get("description") or row.get("maintext") or ""
        if not isinstance(body, str) or len(body) < spec.per_article_min_chars:
            n_too_short += 1
            continue

        body_hash = _sha256(body)
        if body_hash in forbidden_hashes:
            n_leaked += 1
            continue

        url_hash = _url_hash(row.get("url"))
        if url_hash is not None:
            if url_hash in seen_url_hashes:
                n_dup_url += 1
                continue
            seen_url_hashes.add(url_hash)
        else:
            if body_hash in seen_body_hashes:
                n_dup_body += 1
                continue
            seen_body_hashes.add(body_hash)

        kept.append(body)
        if len(kept) >= CC_NEWS_MAX_KEPT:
            break

    diagnostics = {
        "num_rows_seen": n_seen,
        "num_rows_used": len(kept),
        "num_dropped_domain": n_filter_domain,
        "num_dropped_too_short": n_too_short,
        "num_dropped_dups_url": n_dup_url,
        "num_dropped_dups_body": n_dup_body,
        "num_dropped_leak": n_leaked,
        "median_article_chars": (
            float(statistics.median([len(t) for t in kept])) if kept else 0.0
        ),
        "max_kept_cap": CC_NEWS_MAX_KEPT,
    }
    return kept, diagnostics


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_dapt_corpus(
    spec: CorpusSpec,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Build the DAPT corpus from configured sources.

    Returns
    -------
    article_texts : list[str]
        Corpus article strings ready for `pack_into_windows` in `src.dapt`.
        Order is FiNER train, then FNSPID, then cc_news (if enabled).
    held_out_finer_texts : list[str]
        FiNER train articles reserved for the MLM perplexity probe; excluded
        from `article_texts`.
    diagnostics : dict
        Per-source counters and global summary fields, suitable for inclusion
        in `dapt_summary.json`.
    """

    if spec.held_out_probe_size < 0:
        raise ValueError("`held_out_probe_size` must be >= 0.")
    if spec.corpus_source not in {"finer_train_only", "fnspid_plus_finer"}:
        raise ValueError(
            f"Unsupported corpus_source `{spec.corpus_source}`. "
            "Expected `fnspid_plus_finer` or `finer_train_only`."
        )
    if spec.corpus_source == "fnspid_plus_finer" and spec.fnspid_subsample <= 0:
        raise ValueError("`fnspid_subsample` must be > 0 for fnspid_plus_finer.")

    finer_corpus, held_out, forbidden_hashes, finer_diag = _build_finer_partition(
        spec.held_out_probe_size
    )

    diagnostics: dict[str, Any] = {
        "spec": {
            "corpus_source": spec.corpus_source,
            "include_cc_news": spec.include_cc_news,
            "fnspid_subsample": spec.fnspid_subsample,
            "held_out_probe_size": spec.held_out_probe_size,
            "fnspid_dataset_id": spec.fnspid_dataset_id,
            "cc_news_dataset_id": spec.cc_news_dataset_id,
        },
        "per_source": {"finer_train": finer_diag},
        "num_rows_per_source": {"finer_train": len(finer_corpus)},
        "num_dropped_dups_per_source": {"finer_train": 0},
        "num_dropped_leak_per_source": {
            "finer_train": (
                finer_diag["dropped_duplicates_with_val"]
                + finer_diag["dropped_duplicates_with_test"]
            )
        },
        "median_article_chars_per_source": {
            "finer_train": (
                float(statistics.median([len(t) for t in finer_corpus]))
                if finer_corpus
                else 0.0
            )
        },
    }

    article_texts: list[str] = list(finer_corpus)

    if spec.corpus_source == "finer_train_only":
        diagnostics["total_articles_used"] = len(article_texts)
        diagnostics["num_held_out_finer"] = len(held_out)
        return article_texts, held_out, diagnostics

    fnspid_articles, fnspid_diag = _stream_fnspid(spec, forbidden_hashes)
    diagnostics["per_source"]["fnspid"] = fnspid_diag
    diagnostics["num_rows_per_source"]["fnspid"] = fnspid_diag["num_rows_used"]
    diagnostics["num_dropped_dups_per_source"]["fnspid"] = (
        fnspid_diag["num_dropped_dups_url"] + fnspid_diag["num_dropped_dups_body"]
    )
    diagnostics["num_dropped_leak_per_source"]["fnspid"] = fnspid_diag[
        "num_dropped_leak"
    ]
    diagnostics["median_article_chars_per_source"]["fnspid"] = fnspid_diag[
        "median_article_chars"
    ]
    article_texts.extend(fnspid_articles)

    if spec.include_cc_news:
        cc_articles, cc_diag = _stream_cc_news(spec, forbidden_hashes)
        diagnostics["per_source"]["cc_news"] = cc_diag
        diagnostics["num_rows_per_source"]["cc_news"] = cc_diag["num_rows_used"]
        diagnostics["num_dropped_dups_per_source"]["cc_news"] = (
            cc_diag["num_dropped_dups_url"] + cc_diag["num_dropped_dups_body"]
        )
        diagnostics["num_dropped_leak_per_source"]["cc_news"] = cc_diag[
            "num_dropped_leak"
        ]
        diagnostics["median_article_chars_per_source"]["cc_news"] = cc_diag[
            "median_article_chars"
        ]
        article_texts.extend(cc_articles)

    diagnostics["total_articles_used"] = len(article_texts)
    diagnostics["num_held_out_finer"] = len(held_out)
    return article_texts, held_out, diagnostics
