# Integration Design

## Pipeline

    Source1 + Source2 + Source3
                 |
                 v
    Existing blocking.py ranking
                 |
                 v
    Frozen final candidate mapping
                 |
                 v
    Replaceable pairwise matcher
                 |
                 v
    Validated decisions
                 |
                 v
    candidate_pairs.tsv + matching_results.tsv

integration.py calls the existing SQLite-backed blocking functions without editing
or changing their ranking rules. Its generated, immutable mapping is both the exact
candidate set given to the matcher and the content of candidate_pairs.tsv.

## Matcher Interface

The matching-model branch must expose an object or zero-argument factory whose
predict method accepts:

    predict(
        *,
        source1_rows,
        source2_path,
        source3_path,
        candidate_pairs,
    ) -> Mapping[str, Iterable[str]]

Source 2 and Source 3 are supplied as paths so a matcher can stream only records
needed for its candidates. The mapping contains every Source 1 ID. It supports
empty, singleton, and multiple predicted lists; integration rejects duplicates or
any prediction outside that Source 1 entity's frozen candidate list.

Run it with a real matcher:

    py -3 run_pipeline.py --test-dir dataset/test --output-dir output --matcher matching_model:build_matcher

## Output Contract

Both outputs are UTF-8, tab-delimited TSVs with exactly one test Source 1 row:

| File | Header | List separator |
| --- | --- | --- |
| candidate_pairs.tsv | source1_entity_id, candidate_entity_ids | comma |
| matching_results.tsv | source1_entity_id, matched_entity_ids | comma |

An empty match list is valid. Candidate IDs are created only from Source 2/3 records,
and matching output is validated as a strict candidate subset before writing.

## Validation

Candidate-only training validation:

    py -3 evaluate_training.py --train-dir dataset/train --candidates-only

Real matcher validation measures candidate recall, reduction, precision, recall,
macro F0.5, false merges, and zero/singleton/multiple-match behavior:

    py -3 evaluate_training.py --train-dir dataset/train --matcher matching_model:build_matcher

Validate a test submission:

    py -3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids

## Limitation

This branch contains no pairwise matching model. There is intentionally no default
or fake matcher in the CLI. Candidate-only validation measures blocking coverage,
not matching or competition performance.
