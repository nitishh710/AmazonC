"""Produce test submission files with a separately supplied real matcher."""

from __future__ import annotations

import argparse
import sys

from integration import IntegrationError, load_matcher, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--matcher", required=True, help="Matcher object or factory: module:attribute")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    try:
        result = run_pipeline(
            args.test_dir, args.output_dir, load_matcher(args.matcher), batch_size=args.batch_size
        )
    except (IntegrationError, OSError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {result.candidate_pairs_path}")
    print(f"Wrote {result.matching_results_path}")
    print(
        f"Source1={result.source1_count}, targets={result.target_count}, "
        f"candidates={result.candidate_pair_count}, predictions={result.matched_pair_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
