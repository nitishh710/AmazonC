"""Measure candidate coverage, and model metrics when a real matcher is supplied."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from evaluation import evaluate_candidates, evaluate_predictions, read_gold_matches
from integration import (
    IntegrationError,
    discover_source_files,
    generate_final_candidates,
    load_matcher,
    read_source1,
    validate_predictions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--matcher", help="Real matcher object or factory: module:attribute")
    parser.add_argument("--candidates-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    if not args.matcher and not args.candidates_only:
        parser.error("Use --candidates-only or supply --matcher.")
    try:
        source_files = discover_source_files(args.train_dir)
        source1_rows = read_source1(source_files.source1_path)
        generation = generate_final_candidates(
            source1_rows, source_files, batch_size=args.batch_size
        )
        gold = read_gold_matches(Path(args.train_dir) / "train_ground_truth.tsv")
        report: dict[str, object] = {
            "candidate_metrics": asdict(
                evaluate_candidates(
                    source1_rows, generation.candidate_pairs, gold, generation.target_count
                )
            )
        }
        if args.matcher:
            matcher = load_matcher(args.matcher)
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
            report["matching_metrics"] = evaluate_predictions(
                source1_rows, predictions, gold
            ).to_dict()
        else:
            report["matching_metrics"] = None
            report["matching_metrics_note"] = "No real pairwise matcher was supplied."
    except (IntegrationError, OSError, ValueError) as error:
        print(f"Training validation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
