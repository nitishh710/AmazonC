#!/usr/bin/env python3
"""Build and evaluate multi-stage entity-resolution blocking on a train holdout.

Run ``py blocking.py`` from the repository root. The validation source-1 split is
selected deterministically by entity ID; all training source-2/3 rows form the
candidate corpus. No matching model or external data is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "dataset" / "train"
ANALYSIS = ROOT / "analysis"
SUMMARY_OUT = ANALYSIS / "blocking_validation.json"
REPORT_OUT = ROOT / "BLOCKING_ANALYSIS.md"
FIELDS = ("entity_id", "business_name", "business_address", "country")

# Kind values are persisted in SQLite and used to keep the stages interpretable.
EXACT_NAME, NAME_TOKEN, NAME_NGRAM, POSTAL, ADDRESS_TOKEN = range(1, 6)
KIND_NAMES = {
    EXACT_NAME: "exact_name",
    NAME_TOKEN: "name_token",
    NAME_NGRAM: "name_character_trigram",
    POSTAL: "postal_number",
    ADDRESS_TOKEN: "address_token",
}
DF_LIMITS = {
    EXACT_NAME: 500,
    NAME_TOKEN: 3_000,
    NAME_NGRAM: 1_000,
    POSTAL: 5_000,
    ADDRESS_TOKEN: 2_000,
}
MAX_CANDIDATE_LIMIT = 2_000
EVALUATION_LIMITS = (100, 250, 500, 1_000, 2_000)
TOKEN_RE = re.compile(r"[a-z0-9]+")
POSTAL_RE = re.compile(r"\b\d{5,6}\b")


@dataclass(frozen=True)
class Record:
    entity_id: str
    name: str
    address: str
    country: str


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def tokens(text: str, min_length: int = 3) -> set[str]:
    return {token for token in TOKEN_RE.findall(text) if len(token) >= min_length}


def trigrams(text: str) -> set[str]:
    compact = text.replace(" ", "")
    return {compact[i:i + 3] for i in range(max(0, len(compact) - 2))}


def postal_numbers(text: str) -> set[str]:
    return set(POSTAL_RE.findall(text))


def split_bucket(entity_id: str) -> int:
    digest = hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 100


def read_source1(path: Path, validation_bucket: int) -> tuple[list[Record], int]:
    heldout: list[Record] = []
    total = 0
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected schema in {path}: {reader.fieldnames}")
        for row in reader:
            total += 1
            entity_id = row["entity_id"].strip()
            if split_bucket(entity_id) == validation_bucket:
                heldout.append(Record(
                    entity_id,
                    normalize(row["business_name"]),
                    normalize(row["business_address"]),
                    row["country"].strip(),
                ))
    if not heldout:
        raise ValueError(f"Validation bucket {validation_bucket} selected no source-1 records")
    return heldout, total


def read_labels(path: Path, heldout_ids: set[str]) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = ["source1_entity_id", "matched_entity_ids"]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected ground-truth schema in {path}: {reader.fieldnames}")
        for row in reader:
            source1_id = row["source1_entity_id"]
            if source1_id in heldout_ids:
                labels[source1_id] = {value for value in row["matched_entity_ids"].split(",") if value}
    missing = heldout_ids - labels.keys()
    if missing:
        raise ValueError(f"Ground truth is missing {len(missing)} validation source-1 IDs")
    return labels


def query_feature_sets(records: list[Record]) -> dict[int, set[tuple[str, str]]]:
    features: dict[int, set[tuple[str, str]]] = {
        EXACT_NAME: set(),
        NAME_TOKEN: set(),
        NAME_NGRAM: set(),
        POSTAL: set(),
        ADDRESS_TOKEN: set(),
    }
    for record in records:
        features[EXACT_NAME].add((record.country, record.name))
        features[NAME_TOKEN].update((record.country, value) for value in tokens(record.name))
        features[NAME_NGRAM].update((record.country, value) for value in trigrams(record.name))
        features[POSTAL].update((record.country, value) for value in postal_numbers(record.address))
        features[ADDRESS_TOKEN].update(
            (record.country, value) for value in tokens(record.address, min_length=4)
        )
    return features


def increment_relevant(counter: Counter, kind: int, country: str, values: Iterable[str],
                       query_features: dict[int, set[tuple[str, str]]]) -> None:
    relevant = query_features[kind]
    for value in values:
        key = (country, value)
        if key in relevant:
            counter[(kind, *key)] += 1


def target_keys(country: str, name: str, address: str,
                query_features: dict[int, set[tuple[str, str]]],
                frequencies: Counter) -> set[tuple[int, str, str]]:
    keys: set[tuple[int, str, str]] = set()
    candidates = (
        (EXACT_NAME, (name,)),
        (NAME_TOKEN, tokens(name)),
        (NAME_NGRAM, trigrams(name)),
        (POSTAL, postal_numbers(address)),
        (ADDRESS_TOKEN, tokens(address, min_length=4)),
    )
    for kind, values in candidates:
        query_values = query_features[kind]
        for value in values:
            key = (country, value)
            if key in query_values and frequencies[(kind, *key)] <= DF_LIMITS[kind]:
                keys.add((kind, country, value))
    return keys


def key_digest(kind: int, country: str, value: str) -> bytes:
    return hashlib.blake2b(f"{kind}\0{country}\0{value}".encode("utf-8"), digest_size=16).digest()


def load_validation_key_frequencies(target_paths: tuple[Path, Path],
                                    query_features: dict[int, set[tuple[str, str]]],
                                    labels: dict[str, set[str]],
                                    validation_countries: dict[str, str]) -> tuple[Counter, dict]:
    frequencies: Counter = Counter()
    target_to_source1 = {
        target_id: source1_id
        for source1_id, target_ids in labels.items()
        for target_id in target_ids
    }
    seen_label_targets: set[str] = set()
    country_mismatches = 0
    for path in target_paths:
        print(f"Counting validation-relevant keys in {path.name}...", flush=True)
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"Unexpected schema in {path}: {reader.fieldnames}")
            for row in reader:
                name = normalize(row["business_name"].strip())
                address = normalize(row["business_address"].strip())
                country = row["country"].strip()
                increment_relevant(frequencies, EXACT_NAME, country, (name,), query_features)
                increment_relevant(frequencies, NAME_TOKEN, country, tokens(name), query_features)
                increment_relevant(frequencies, NAME_NGRAM, country, trigrams(name), query_features)
                increment_relevant(frequencies, POSTAL, country, postal_numbers(address), query_features)
                increment_relevant(
                    frequencies, ADDRESS_TOKEN, country,
                    tokens(address, min_length=4), query_features,
                )
                target_id = row["entity_id"].strip()
                source1_id = target_to_source1.get(target_id)
                if source1_id is not None:
                    seen_label_targets.add(target_id)
                    country_mismatches += country != validation_countries[source1_id]
    diagnostics = {
        "ground_truth_target_ids_found": len(seen_label_targets),
        "ground_truth_target_country_mismatches": country_mismatches,
    }
    return frequencies, diagnostics


def build_target_index(db: sqlite3.Connection, target_paths: tuple[Path, Path],
                       query_features: dict[int, set[tuple[str, str]]],
                       frequencies: Counter) -> int:
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("CREATE TABLE blocks (k BLOB NOT NULL, target_id TEXT NOT NULL, kind INTEGER NOT NULL, PRIMARY KEY (k, target_id)) WITHOUT ROWID")
    insert = "INSERT OR IGNORE INTO blocks(k, target_id, kind) VALUES (?, ?, ?)"
    batch: list[tuple[bytes, str, int]] = []
    target_count = 0
    for path in target_paths:
        print(f"Indexing candidates from {path.name}...", flush=True)
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"Unexpected schema in {path}: {reader.fieldnames}")
            for row in reader:
                target_count += 1
                target_id = row["entity_id"].strip()
                name = normalize(row["business_name"].strip())
                address = normalize(row["business_address"].strip())
                country = row["country"].strip()
                for kind, key_country, value in target_keys(
                    country, name, address, query_features, frequencies
                ):
                    batch.append((key_digest(kind, key_country, value), target_id, kind))
                if len(batch) >= 100_000:
                    db.executemany(insert, batch)
                    db.commit()
                    batch.clear()
        if batch:
            db.executemany(insert, batch)
            db.commit()
            batch.clear()
    db.execute("CREATE INDEX blocks_by_target ON blocks(target_id)")
    db.commit()
    return target_count


def key_weight(kind: int, df: int) -> float:
    idf = 1.0 + math.log1p(DF_LIMITS[kind] / max(df, 1))
    if kind == EXACT_NAME:
        return 8.0 + idf
    if kind == NAME_TOKEN:
        return 2.0 * idf
    if kind == NAME_NGRAM:
        return 0.5 * idf
    if kind == POSTAL:
        return 1.5 * idf
    return idf


def query_key_rows(batch: list[Record], start_index: int,
                   query_features: dict[int, set[tuple[str, str]]],
                   frequencies: Counter) -> list[tuple[int, bytes, int, float]]:
    rows = []
    for offset, record in enumerate(batch):
        qid = start_index + offset
        for kind, country, value in target_keys(
            record.country, record.name, record.address, query_features, frequencies
        ):
            df = frequencies[(kind, country, value)]
            rows.append((qid, key_digest(kind, country, value), kind, key_weight(kind, df)))
    return rows


def rank_candidates(db: sqlite3.Connection, records: list[Record], labels: dict[str, set[str]],
                    query_features: dict[int, set[tuple[str, str]]], frequencies: Counter,
                    batch_size: int = 500) -> tuple[dict[str, dict], int]:
    db.execute("CREATE TEMP TABLE query_keys (qid INTEGER, k BLOB, kind INTEGER, weight REAL)")
    db.execute("CREATE INDEX query_keys_by_key ON query_keys(k)")
    query = """
        WITH scored AS (
            SELECT q.qid, b.target_id,
                   SUM(CASE WHEN q.kind = 1 THEN q.weight ELSE 0 END) AS exact_score,
                   SUM(CASE WHEN q.kind IN (1, 2, 3) THEN q.weight ELSE 0 END) AS name_score,
                   SUM(CASE WHEN q.kind IN (4, 5) THEN q.weight ELSE 0 END) AS address_score
            FROM query_keys q JOIN blocks b ON b.k = q.k
            GROUP BY q.qid, b.target_id
        ), weighted AS (
            SELECT qid, target_id, exact_score, name_score, address_score,
                   (2.0 * name_score + address_score) AS full_score
            FROM scored
        ), ranked AS (
            SELECT qid, target_id, exact_score, name_score, full_score,
                   CASE WHEN exact_score > 0 THEN
                       ROW_NUMBER() OVER (PARTITION BY qid ORDER BY exact_score DESC, target_id)
                   END AS exact_rank,
                   CASE WHEN name_score > 0 THEN
                       ROW_NUMBER() OVER (PARTITION BY qid ORDER BY name_score DESC, exact_score DESC, target_id)
                   END AS name_rank,
                   ROW_NUMBER() OVER (PARTITION BY qid ORDER BY full_score DESC, name_score DESC, target_id) AS full_rank
            FROM weighted
        )
        SELECT qid, target_id, exact_rank, name_rank, full_rank
        FROM ranked
        WHERE exact_rank <= ? OR name_rank <= ? OR full_rank <= ?
    """
    result: dict[str, dict] = {}
    total_block_rows = 0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        rows = query_key_rows(batch, start, query_features, frequencies)
        db.execute("DELETE FROM query_keys")
        db.executemany("INSERT INTO query_keys VALUES (?, ?, ?, ?)", rows)
        db.commit()
        total_block_rows += db.execute("SELECT COUNT(*) FROM query_keys").fetchone()[0]
        for qid, target_id, exact_rank, name_rank, full_rank in db.execute(
            query, (MAX_CANDIDATE_LIMIT, MAX_CANDIDATE_LIMIT, MAX_CANDIDATE_LIMIT)
        ):
            result.setdefault(records[qid].entity_id, {})[target_id] = (exact_rank, name_rank, full_rank)
        print(f"Ranked validation queries {start + len(batch):,}/{len(records):,}", flush=True)
    return result, total_block_rows


def calculate_metrics(records: list[Record], labels: dict[str, set[str]],
                      ranked: dict[str, dict[str, tuple[int, int, int]]],
                      target_count: int) -> dict:
    pair_total = len(records) * target_count
    limits: dict[str, dict] = {}
    for label, rank_index in (("exact_name", 0), ("name_only", 1), ("full_multistage", 2)):
        by_limit = {}
        for limit in EVALUATION_LIMITS:
            found_links = total_candidates = 0
            per_entity_recall = []
            counts = []
            for record in records:
                candidates = ranked.get(record.entity_id, {})
                candidates_at_limit = {
                    target_id for target_id, ranks in candidates.items()
                    if ranks[rank_index] is not None and ranks[rank_index] <= limit
                }
                true_targets = labels[record.entity_id]
                found = len(candidates_at_limit & true_targets)
                total_candidates += len(candidates_at_limit)
                found_links += found
                counts.append(len(candidates_at_limit))
                if true_targets:
                    per_entity_recall.append(found / len(true_targets))
            links_total = sum(map(len, labels.values()))
            by_limit[str(limit)] = {
                "candidate_recall_micro": found_links / links_total if links_total else 0.0,
                "mean_recall_per_linked_source1": sum(per_entity_recall) / len(per_entity_recall) if per_entity_recall else 0.0,
                "average_candidates_per_source1": total_candidates / len(records),
                "candidate_reduction_ratio": 1.0 - total_candidates / pair_total if pair_total else 0.0,
                "candidate_count_p50": percentile(counts, 0.50),
                "candidate_count_p95": percentile(counts, 0.95),
                "candidate_count_p99": percentile(counts, 0.99),
                "candidate_count_max": max(counts, default=0),
            }
        limits[label] = by_limit

    singleton_count = sum(not labels[record.entity_id] for record in records)
    country_counts = Counter(record.country for record in records)
    return {
        "records": len(records),
        "positive_source1_records": len(records) - singleton_count,
        "singleton_source1_records": singleton_count,
        "singleton_rate": singleton_count / len(records),
        "ground_truth_links": sum(map(len, labels.values())),
        "country_distribution": dict(country_counts),
        "tiers": limits,
    }


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def write_report(summary: dict) -> None:
    default_tier = summary["metrics"]["tiers"]["full_multistage"]
    lines = [
        "# Blocking Strategy and Validation",
        "",
        "## Scope and validation design",
        "",
        "This stage implements candidate generation only; it does not train a matching model or predict final matches. It uses only the supplied training TSVs.",
        "",
        f"Validation source-1 entities are the deterministic 1-in-100 BLAKE2b hash bucket `{summary['validation_bucket']}` ({summary['metrics']['records']:,} of {summary['training_source1_rows']:,}). The remaining source-1 rows are the conceptual training partition. All training Source 2 and Source 3 rows are retained as the candidate corpus because those records are the search universe, and source-1 rows are not candidates.",
        "",
        "## Multi-stage blocking",
        "",
        "1. **Exact normalized name:** lowercase alphanumeric normalization; only values with at most 500 target postings are indexed.",
        "2. **Name token overlap:** shared tokens of at least three characters, retaining only corpus document frequencies up to 3,000.",
        "3. **Name character trigrams:** shared trigrams from names, capped at 1,000 target postings per key to tolerate small spelling edits without unbounded expansions.",
        "4. **Address evidence:** shared 5–6 digit address number (up to 5,000 postings) or address token of at least four characters (up to 2,000 postings).",
        "",
        "All block keys are scoped by the record's exact country label. This uses country equality to reduce cross-country collisions without hard-coding a country list, so unseen labels remain eligible. A deterministic evidence score ranks candidates using twice the name-evidence score plus address-evidence score; exact name, rare shared name tokens, character trigrams, postal-like numbers, and address tokens contribute. Scores are only for candidate ordering, not match decisions. For evaluation, each tier is independently capped at 100/250/500/1,000/2,000 candidates per S1.",
        "",
        "## Held-out blocking results",
        "",
        f"The validation set contains {summary['metrics']['ground_truth_links']:,} target links and {summary['metrics']['singleton_source1_records']:,} singletons ({summary['metrics']['singleton_rate']:.2%}). Country counts: {', '.join(f'{country} {count:,}' for country, count in summary['metrics']['country_distribution'].items())}.",
        f"The index found {summary['country_diagnostics']['ground_truth_target_ids_found']:,} of the validation target IDs in the candidate corpus; {summary['country_diagnostics']['ground_truth_target_country_mismatches']:,} of those labels cross a country boundary.",
        "",
        "| Candidate tier | Cap/S1 | Link candidate recall | Mean recall per linked S1 | Avg candidates/S1 | Reduction ratio | Candidate count p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tier, values in summary["metrics"]["tiers"].items():
        for cap, metrics in values.items():
            lines.append(
                f"| {tier.replace('_', ' ')} | {cap} | {metrics['candidate_recall_micro']:.4%} | "
                f"{metrics['mean_recall_per_linked_source1']:.4%} | {metrics['average_candidates_per_source1']:.2f} | "
                f"{metrics['candidate_reduction_ratio']:.6%} | {metrics['candidate_count_p95']:,} |"
            )
    lines += [
        "",
        f"At the largest evaluated full multistage cap of {MAX_CANDIDATE_LIMIT}, average candidates are {default_tier[str(MAX_CANDIDATE_LIMIT)]['average_candidates_per_source1']:.2f} per validation S1 (p99 {default_tier[str(MAX_CANDIDATE_LIMIT)]['candidate_count_p99']:,}; maximum {default_tier[str(MAX_CANDIDATE_LIMIT)]['candidate_count_max']:,}). Candidate recall is measured over ground-truth links; singleton rows have no positive links and are excluded from the per-linked-S1 recall mean.",
        "",
        "## Reproduction and outputs",
        "",
        "Run `py blocking.py` from the repository root. The script rebuilds the temporary SQLite postings index from the supplied training Source 2/3 data, evaluates the deterministic held-out Source 1 bucket, removes its temporary index, and writes [`analysis/blocking_validation.json`](./analysis/blocking_validation.json) plus this report. Use `--bucket` and `--batch-size` to select another deterministic bucket or adjust query batching. No external lookup or data is used.",
        "",
        "This is a candidate-generation baseline, not a matching model. The score used to order candidates must not be treated as a final match probability or threshold.",
        "",
    ]
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", type=int, default=17, choices=range(100),
                        help="BLAKE2b validation bucket in [0, 99] (default: 17)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="number of validation source-1 queries processed at once")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    ANALYSIS.mkdir(exist_ok=True)
    validation, source1_total = read_source1(TRAIN / "train_source1.tsv", args.bucket)
    validation_ids = {record.entity_id for record in validation}
    labels = read_labels(TRAIN / "train_ground_truth.tsv", validation_ids)
    features = query_feature_sets(validation)
    target_paths = (TRAIN / "train_source2.tsv", TRAIN / "train_source3.tsv")
    validation_countries = {record.entity_id: record.country for record in validation}
    frequencies, country_diagnostics = load_validation_key_frequencies(
        target_paths, features, labels, validation_countries
    )
    eligible_key_counts = {
        KIND_NAMES[kind]: sum(1 for (key_kind, _, _), count in frequencies.items()
                              if key_kind == kind and count <= DF_LIMITS[kind])
        for kind in KIND_NAMES
    }

    temp = tempfile.NamedTemporaryFile(prefix="blocking-", suffix=".sqlite", dir=ANALYSIS, delete=False)
    temp_path = Path(temp.name)
    temp.close()
    try:
        db = sqlite3.connect(temp_path)
        try:
            target_count = build_target_index(db, target_paths, features, frequencies)
            ranked, query_key_count = rank_candidates(
                db, validation, labels, features, frequencies, batch_size=args.batch_size
            )
            metrics = calculate_metrics(validation, labels, ranked, target_count)
            summary = {
                "validation_bucket": args.bucket,
                "validation_hash": "blake2b-64(entity_id) modulo 100",
                "training_source1_rows": source1_total,
                "validation_fraction": len(validation) / source1_total,
                "target_rows": target_count,
                "candidate_key_postings": db.execute("SELECT COUNT(*) FROM blocks").fetchone()[0],
                "query_key_rows": query_key_count,
                "country_diagnostics": country_diagnostics,
                "eligible_query_keys_by_kind": eligible_key_counts,
                "df_limits": {KIND_NAMES[kind]: limit for kind, limit in DF_LIMITS.items()},
                "max_candidate_limit": MAX_CANDIDATE_LIMIT,
                "metrics": metrics,
            }
        finally:
            db.close()
    finally:
        temp_path.unlink(missing_ok=True)

    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    print(f"Wrote {SUMMARY_OUT}", flush=True)
    print(f"Wrote {REPORT_OUT}", flush=True)
    print(json.dumps(summary["metrics"]["tiers"], indent=2), flush=True)


if __name__ == "__main__":
    main()
