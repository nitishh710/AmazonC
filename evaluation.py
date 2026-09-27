"""Training metrics for blocking candidates and a plugged-in pairwise matcher."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from blocking import Record


@dataclass(frozen=True)
class CandidateMetrics:
    candidate_recall: float
    covered_gold_pairs: int
    gold_pairs: int
    average_candidates_per_source1: float
    candidate_reduction: float
    zero_match_source1_count: int
    zero_match_with_zero_candidates: int
    singleton_source1_count: int
    singleton_candidate_recall: float
    multiple_match_source1_count: int
    multiple_candidate_recall: float


@dataclass(frozen=True)
class GroupMetrics:
    source1_count: int
    exact_set_count: int
    predicted_empty_count: int
    false_merges: int


@dataclass(frozen=True)
class MatchingMetrics:
    precision: float
    recall: float
    macro_f0_5: float
    false_merges: int
    zero_match: GroupMetrics
    singleton: GroupMetrics
    multiple_match: GroupMetrics

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def read_gold_matches(path: str | Path) -> dict[str, set[str]]:
    """Read the challenge's one-row-per-Source1, comma-separated label file."""
    matches: dict[str, set[str]] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"Unexpected label schema: {reader.fieldnames}")
        for row in reader:
            source1_id = row["source1_entity_id"].strip()
            values = row["matched_entity_ids"].strip()
            matches[source1_id] = {value for value in values.split(",") if value}
    return matches


def evaluate_candidates(
    source1_rows: Sequence[Record],
    candidates: Mapping[str, Sequence[str]],
    gold_matches: Mapping[str, set[str]],
    target_count: int,
) -> CandidateMetrics:
    """Measure blocking coverage and final-candidate reduction."""
    counts = {"zero": 0, "single": 0, "multiple": 0}
    covered = {"single": 0, "multiple": 0}
    gold_total = gold_covered = candidate_total = zero_empty = 0
    for record in source1_rows:
        gold = gold_matches.get(record.entity_id, set())
        selected = set(candidates[record.entity_id])
        hits = len(gold & selected)
        gold_total += len(gold)
        gold_covered += hits
        candidate_total += len(selected)
        if not gold:
            counts["zero"] += 1
            zero_empty += not selected
        elif len(gold) == 1:
            counts["single"] += 1
            covered["single"] += hits
        else:
            counts["multiple"] += 1
            covered["multiple"] += hits
    multiple_gold = sum(
        len(gold_matches.get(record.entity_id, set()))
        for record in source1_rows
        if len(gold_matches.get(record.entity_id, set())) > 1
    )
    return CandidateMetrics(
        _ratio(gold_covered, gold_total),
        gold_covered,
        gold_total,
        _ratio(candidate_total, len(source1_rows)),
        1.0 - _ratio(candidate_total, len(source1_rows) * target_count),
        counts["zero"],
        zero_empty,
        counts["single"],
        _ratio(covered["single"], counts["single"]),
        counts["multiple"],
        _ratio(covered["multiple"], multiple_gold),
    )


def evaluate_predictions(
    source1_rows: Sequence[Record],
    predictions: Mapping[str, Sequence[str]],
    gold_matches: Mapping[str, set[str]],
) -> MatchingMetrics:
    """Measure pair precision/recall, macro F0.5, false merges, and group behavior."""
    true_positives = predicted_total = gold_total = false_merges = 0
    scores: list[float] = []
    groups: dict[str, list[tuple[set[str], set[str]]]] = {"zero": [], "single": [], "multiple": []}
    for record in source1_rows:
        gold = gold_matches.get(record.entity_id, set())
        predicted = set(predictions.get(record.entity_id, ()))
        true_positives += len(gold & predicted)
        predicted_total += len(predicted)
        gold_total += len(gold)
        false_merges += len(predicted - gold)
        scores.append(_f_beta(gold, predicted))
        groups["zero" if not gold else "single" if len(gold) == 1 else "multiple"].append((gold, predicted))
    return MatchingMetrics(
        _ratio(true_positives, predicted_total, empty_value=1.0),
        _ratio(true_positives, gold_total),
        _ratio(sum(scores), len(scores)),
        false_merges,
        _group(groups["zero"]),
        _group(groups["single"]),
        _group(groups["multiple"]),
    )


def _group(values: Sequence[tuple[set[str], set[str]]]) -> GroupMetrics:
    return GroupMetrics(
        len(values),
        sum(gold == predicted for gold, predicted in values),
        sum(not predicted for _, predicted in values),
        sum(len(predicted - gold) for gold, predicted in values),
    )


def _f_beta(gold: set[str], predicted: set[str]) -> float:
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0
    precision = _ratio(len(gold & predicted), len(predicted))
    recall = _ratio(len(gold & predicted), len(gold))
    return 1.25 * precision * recall / (0.25 * precision + recall) if precision + recall else 0.0


def _ratio(numerator: int | float, denominator: int | float, empty_value: float = 0.0) -> float:
    return numerator / denominator if denominator else empty_value
