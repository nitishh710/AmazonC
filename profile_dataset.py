#!/usr/bin/env python3
"""Profile the supplied entity-resolution TSVs without building a matcher.

Run ``py profile_dataset.py`` from the repository root. It reads only local
challenge files and writes ``analysis/profiling_summary.json``. Sources are
processed one at a time. Frequency keys are 128-bit BLAKE2b digests of
normalised text, making full-file duplicate summaries feasible in memory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "dataset", ROOT / "analysis" / "profiling_summary.json"
FIELDS = ("entity_id", "business_name", "business_address", "country")
MISSING = {"", "na", "n/a", "null", "none", "nan", "-"}
LEN_BINS = (0, 1, 5, 10, 20, 40, 80, 160)


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def duplicate_summary(counts: Counter) -> dict[str, int]:
    groups = [n for n in counts.values() if n > 1]
    return {"distinct": len(counts), "duplicate_groups": len(groups),
            "rows_in_duplicate_groups": sum(groups), "duplicate_rows_beyond_first": sum(n - 1 for n in groups),
            "largest_group": max(counts.values(), default=0)}


def hist(counts: Counter) -> dict[str, int]:
    answer = Counter()
    for length, n in counts.items():
        for lo, hi in zip(LEN_BINS, LEN_BINS[1:]):
            if lo <= length < hi:
                answer[f"{lo}-{hi - 1}"] += n
                break
        else:
            answer[f"{LEN_BINS[-1]}+"] += n
    return dict(answer)


def profile_source(path: Path) -> dict:
    rows, missing, countries, features = 0, Counter(), Counter(), Counter()
    lengths = {"business_name": Counter(), "business_address": Counter()}
    # Keep only the ID and name maps in this pass.  A second pass handles address
    # and whole-record frequencies: holding all three large maps at once can exceed
    # the memory available for these million-row TSVs.
    ids, names = set(), Counter()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected schema in {path}: {reader.fieldnames}")
        for row in reader:
            rows += 1
            values = {k: row[k].strip() for k in FIELDS}
            ids.add(values["entity_id"])
            for field, value in values.items():
                if value.lower() in MISSING:
                    missing[field] += 1
            countries[values["country"] or "<blank>"] += 1
            name, address = values["business_name"], values["business_address"]
            name_n, address_n = norm(name), norm(address)
            names[digest(name_n)] += 1
            lengths["business_name"][len(name)] += 1; lengths["business_address"][len(address)] += 1
            features["names_with_digits"] += bool(re.search(r"\d", name))
            features["names_with_punctuation"] += bool(re.search(r"[^\w\s]", name))
            features["names_with_ampersand_or_plus"] += bool(re.search(r"[&+]", name))
            features["addresses_with_digits"] += bool(re.search(r"\d", address))
            features["addresses_with_comma"] += "," in address
            features["addresses_with_postal_like_number"] += bool(re.search(r"\b\d{5,6}\b", address))
    addresses = Counter()
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            addresses[digest(norm(row["business_address"].strip()))] += 1
    records = Counter()
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = "\x1f".join((norm(row["business_name"].strip()), norm(row["business_address"].strip()), norm(row["country"].strip())))
            records[digest(key)] += 1
    return {"rows": rows, "unique_entity_ids": len(ids), "duplicate_entity_ids": rows-len(ids),
            "missing_values": dict(missing), "country_distribution": dict(countries.most_common()),
            "length_distributions": {k: hist(v) for k, v in lengths.items()},
            "field_duplicate_patterns_normalised": {"business_name": duplicate_summary(names), "business_address": duplicate_summary(addresses)},
            "exact_duplicate_patterns_all_non_id_fields": duplicate_summary(records), "characteristics": dict(features)}


def profile_ground_truth(path: Path, source1_count: int) -> dict:
    rows, links, singletons = 0, 0, 0
    links_per_s1, source_combos, targets, by_source = Counter(), Counter(), Counter(), Counter()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"Unexpected ground-truth schema: {reader.fieldnames}")
        for row in reader:
            rows += 1; matches = [x for x in row["matched_entity_ids"].split(",") if x]
            links += len(matches); links_per_s1[len(matches)] += 1
            if not matches: singletons += 1
            source_combos["+".join(sorted({x[:2] for x in matches})) if matches else "none"] += 1
            for target in matches:
                targets[digest(target)] += 1; by_source[target[:2]] += 1
    return {"ground_truth_rows": rows, "source1_records_without_ground_truth_row": source1_count-rows,
            "total_links": links, "singleton_source1_records": singletons, "singleton_rate": singletons/source1_count,
            "links_per_source1_distribution": dict(sorted(links_per_s1.items())),
            "source_pair_statistics_by_source1": dict(source_combos), "link_counts_by_target_source": dict(by_source),
            "target_reuse": duplicate_summary(targets)}


def main() -> None:
    sources = {}
    for split in ("train", "test"):
        for number in (1, 2, 3):
            key = f"{split}_S{number}"; path = DATA / split / f"{split}_source{number}.tsv"
            print(f"Profiling {key}...", flush=True); sources[key] = profile_source(path)
    print("Profiling train ground truth...", flush=True)
    ground_truth = profile_ground_truth(DATA / "train" / "train_ground_truth.tsv", sources["train_S1"]["rows"])
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps({"sources": sources, "ground_truth": ground_truth}, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
