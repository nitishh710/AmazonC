from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from blocking import Record
from evaluation import evaluate_candidates, evaluate_predictions
from integration import IntegrationError, run_pipeline, validate_predictions


class FirstCandidateMatcher:
    def predict(self, *, source1_rows, source2_path, source3_path, candidate_pairs):
        return {"S1-1": (candidate_pairs["S1-1"][0],)}


class IntegrationTests(unittest.TestCase):
    def test_pipeline_writes_complete_submission_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "test"
            data_dir.mkdir()
            header = ["entity_id", "business_name", "business_address", "country"]
            data = {
                "test_source1.tsv": [["S1-1", "Acme", "1 Main Street", "US"], ["S1-2", "None", "", "US"]],
                "test_source2.tsv": [["S2-1", "Acme", "1 Main Street", "US"]],
                "test_source3.tsv": [["S3-1", "Other", "9 Lane", "US"]],
            }
            for filename, rows in data.items():
                with (data_dir / filename).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle, delimiter="\t")
                    writer.writerow(header)
                    writer.writerows(rows)
            result = run_pipeline(data_dir, Path(temporary) / "output", FirstCandidateMatcher())
            with result.matching_results_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["source1_entity_id"] for row in rows], ["S1-1", "S1-2"])
            self.assertEqual(rows[0]["matched_entity_ids"], "S2-1")
            self.assertEqual(rows[1]["matched_entity_ids"], "")

    def test_prediction_outside_candidates_is_rejected(self):
        with self.assertRaises(IntegrationError):
            validate_predictions(
                [Record("S1-1", "", "", "")], {"S1-1": ("S2-1",)}, {"S1-1": ("S3-1",)}
            )

    def test_metrics_cover_zero_singleton_and_multiple_matches(self):
        rows = [Record("S1-0", "", "", ""), Record("S1-1", "", "", ""), Record("S1-2", "", "", "")]
        gold = {"S1-0": set(), "S1-1": {"S2-1"}, "S1-2": {"S2-2", "S3-1"}}
        candidates = {"S1-0": (), "S1-1": ("S2-1",), "S1-2": ("S2-2", "S3-1", "S3-2")}
        candidate_metrics = evaluate_candidates(rows, candidates, gold, 4)
        self.assertEqual(candidate_metrics.candidate_recall, 1.0)
        self.assertEqual(candidate_metrics.zero_match_with_zero_candidates, 1)
        metrics = evaluate_predictions(
            rows, {"S1-0": (), "S1-1": ("S2-1",), "S1-2": ("S2-2", "S3-2")}, gold
        )
        self.assertEqual(metrics.false_merges, 1)
        self.assertAlmostEqual(metrics.precision, 2 / 3)
        self.assertAlmostEqual(metrics.recall, 2 / 3)
        self.assertAlmostEqual(metrics.macro_f0_5, 5 / 6)


if __name__ == "__main__":
    unittest.main()
