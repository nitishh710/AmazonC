"""Blocking-to-matcher integration and strict submission-file writing."""

from __future__ import annotations

import csv
import importlib
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from blocking import (
    Record,
    build_target_index,
    load_validation_key_frequencies,
    normalize,
    query_feature_sets,
    rank_candidates,
)

FIELDS = ("entity_id", "business_name", "business_address", "country")


class IntegrationError(ValueError):
    """Raised when an integration component breaks the submission contract."""


@dataclass(frozen=True)
class SourceFiles:
    source1_path: Path
    source2_path: Path
    source3_path: Path


@dataclass(frozen=True)
class CandidateGeneration:
    candidate_pairs: Mapping[str, tuple[str, ...]]
    target_count: int


@dataclass(frozen=True)
class PipelineResult:
    candidate_pairs_path: Path
    matching_results_path: Path
    source1_count: int
    target_count: int
    candidate_pair_count: int
    matched_pair_count: int


@runtime_checkable
class CandidateMatcher(Protocol):
    """Contract for the separately developed pairwise matcher."""

    def predict(
        self,
        *,
        source1_rows: Sequence[Record],
        source2_path: Path,
        source3_path: Path,
        candidate_pairs: Mapping[str, Sequence[str]],
    ) -> Mapping[str, Iterable[str]]:
        """Return zero, one, or many target IDs per Source 1 ID."""


def discover_source_files(data_dir: str | Path) -> SourceFiles:
    """Find the train_* or test_* source TSVs in a supplied data directory."""
    directory = Path(data_dir)
    prefixes = ("test", "train")
    for prefix in prefixes:
        paths = tuple(directory / f"{prefix}_source{number}.tsv" for number in (1, 2, 3))
        if all(path.is_file() for path in paths):
            return SourceFiles(*paths)
    raise IntegrationError(
        f"Expected test_source1/2/3.tsv or train_source1/2/3.tsv in {directory}."
    )


def read_source1(path: str | Path) -> list[Record]:
    """Load and normalize Source 1 using the same fields as blocking.py."""
    records: list[Record] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise IntegrationError(f"Unexpected schema in {path}: {reader.fieldnames}")
        for row in reader:
            entity_id = row["entity_id"].strip()
            if not entity_id:
                raise IntegrationError(f"Blank Source 1 entity ID in {path}.")
            records.append(
                Record(
                    entity_id,
                    normalize(row["business_name"].strip()),
                    normalize(row["business_address"].strip()),
                    row["country"].strip(),
                )
            )
    if len({record.entity_id for record in records}) != len(records):
        raise IntegrationError("Source1 contains duplicate entity IDs.")
    return records


def generate_final_candidates(
    source1_rows: Sequence[Record],
    source_files: SourceFiles,
    *,
    batch_size: int = 500,
) -> CandidateGeneration:
    """Reuse blocking.py's ranking logic to produce final matcher candidates."""
    if batch_size < 1:
        raise IntegrationError("batch_size must be positive.")

    features = query_feature_sets(list(source1_rows))
    target_paths = (source_files.source2_path, source_files.source3_path)
    frequencies, _ = load_validation_key_frequencies(target_paths, features, {}, {})
    temporary = tempfile.NamedTemporaryFile(prefix="integration-blocking-", suffix=".sqlite", delete=False)
    database_path = Path(temporary.name)
    temporary.close()

    try:
        database = sqlite3.connect(database_path)
        try:
            target_count = build_target_index(database, target_paths, features, frequencies)
            ranked, _ = rank_candidates(
                database, list(source1_rows), {}, features, frequencies, batch_size=batch_size
            )
        finally:
            database.close()
    finally:
        database_path.unlink(missing_ok=True)

    candidates = {
        record.entity_id: tuple(sorted(ranked.get(record.entity_id, {})))
        for record in source1_rows
    }
    _validate_candidate_ids(candidates)
    return CandidateGeneration(MappingProxyType(candidates), target_count)


def validate_predictions(
    source1_rows: Sequence[Record],
    candidate_pairs: Mapping[str, Sequence[str]],
    predictions: Mapping[str, Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    """Validate matcher output and complete omitted Source 1 IDs with empty lists."""
    source1_ids = tuple(record.entity_id for record in source1_rows)
    unknown_source1_ids = set(predictions) - set(source1_ids)
    if unknown_source1_ids:
        raise IntegrationError(
            "Matcher returned unknown Source1 IDs: " + ", ".join(sorted(unknown_source1_ids)[:5])
        )

    validated: dict[str, tuple[str, ...]] = {}
    for source1_id in source1_ids:
        raw_matches = predictions.get(source1_id, ())
        if isinstance(raw_matches, str):
            raise IntegrationError(f"Matcher returned a string instead of IDs for {source1_id!r}.")
        allowed = set(candidate_pairs[source1_id])
        matches: list[str] = []
        seen: set[str] = set()
        for matched_id in raw_matches:
            matched_id = str(matched_id).strip()
            if not matched_id or matched_id in seen:
                raise IntegrationError(f"Invalid duplicate or blank match for {source1_id!r}.")
            if matched_id not in allowed:
                raise IntegrationError(
                    f"Match {matched_id!r} for {source1_id!r} was not a final candidate."
                )
            seen.add(matched_id)
            matches.append(matched_id)
        validated[source1_id] = tuple(matches)
    return validated


def run_pipeline(
    data_dir: str | Path,
    output_dir: str | Path,
    matcher: CandidateMatcher,
    *,
    batch_size: int = 500,
) -> PipelineResult:
    """Generate, freeze, match, validate, and write the two test submissions."""
    if not isinstance(matcher, CandidateMatcher):
        raise IntegrationError("Matcher must implement predict(...).")
    source_files = discover_source_files(data_dir)
    source1_rows = read_source1(source_files.source1_path)
    generation = generate_final_candidates(source1_rows, source_files, batch_size=batch_size)
    predictions = validate_predictions(
        source1_rows,
        generation.candidate_pairs,
        matcher.predict(
            source1_rows=source1_rows,
            source2_path=source_files.source2_path,
            source3_path=source_files.source3_path,
            candidate_pairs=generation.candidate_pairs,
        ),
    )

    destination = Path(output_dir)
    candidate_path = destination / "candidate_pairs.tsv"
    matching_path = destination / "matching_results.tsv"
    _write_submission(
        candidate_path,
        "candidate_entity_ids",
        source1_rows,
        generation.candidate_pairs,
    )
    _write_submission(matching_path, "matched_entity_ids", source1_rows, predictions)
    return PipelineResult(
        candidate_path,
        matching_path,
        len(source1_rows),
        generation.target_count,
        sum(len(values) for values in generation.candidate_pairs.values()),
        sum(len(values) for values in predictions.values()),
    )


def load_matcher(specification: str) -> CandidateMatcher:
    """Load a matcher object, class, or zero-argument factory from module:attribute."""
    if ":" not in specification:
        raise IntegrationError("Matcher must use module:attribute syntax.")
    module_name, attribute_name = specification.split(":", 1)
    try:
        value = getattr(importlib.import_module(module_name), attribute_name)
    except (ImportError, AttributeError) as error:
        raise IntegrationError(f"Could not load matcher {specification!r}: {error}") from error
    if isinstance(value, type):
        value = value()
    elif not isinstance(value, CandidateMatcher) and callable(value):
        value = value()
    if not isinstance(value, CandidateMatcher):
        raise IntegrationError(f"Matcher {specification!r} does not implement predict(...).")
    return value


def _validate_candidate_ids(candidate_pairs: Mapping[str, Sequence[str]]) -> None:
    for source1_id, candidate_ids in candidate_pairs.items():
        if len(candidate_ids) != len(set(candidate_ids)):
            raise IntegrationError(f"Duplicate candidate IDs for {source1_id!r}.")
        invalid = [candidate_id for candidate_id in candidate_ids if not candidate_id.startswith(("S2-", "S3-"))]
        if invalid:
            raise IntegrationError(f"Invalid target ID for {source1_id!r}: {invalid[0]!r}.")


def _write_submission(
    path: Path,
    values_column: str,
    source1_rows: Sequence[Record],
    values_by_source1: Mapping[str, Sequence[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("source1_entity_id", values_column))
        for record in source1_rows:
            writer.writerow((record.entity_id, ",".join(values_by_source1[record.entity_id])))
