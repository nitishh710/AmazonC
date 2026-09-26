# Blocking Strategy and Validation

## Scope and validation design

This stage implements candidate generation only; it does not train a matching model or predict final matches. It uses only the supplied training TSVs.

Validation source-1 entities are the deterministic 1-in-100 BLAKE2b hash bucket `17` (22,264 of 2,206,821). The remaining source-1 rows are the conceptual training partition. All training Source 2 and Source 3 rows are retained as the candidate corpus because those records are the search universe, and source-1 rows are not candidates.

## Multi-stage blocking

1. **Exact normalized name:** lowercase alphanumeric normalization; only values with at most 500 target postings are indexed.
2. **Name token overlap:** shared tokens of at least three characters, retaining only corpus document frequencies up to 1,000.
3. **Name character trigrams:** shared trigrams from names, capped at 150 target postings per key to tolerate small spelling edits without broad expansions.
4. **Address evidence:** shared 5–6 digit address number (up to 5,000 postings) or address token of at least four characters (up to 500 postings).

All stages are unioned, and a deterministic evidence score ranks candidates: exact name dominates, then rare shared name tokens, character trigrams, postal-like numbers, and address tokens. Scores are only for candidate ordering, not match decisions. Country is not used as a hard filter, preserving open-set country coverage. For evaluation, each tier is independently capped at 25/50/100/200 candidates per S1.

## Held-out blocking results

The validation set contains 76,838 target links and 1,231 singletons (5.53%). Country counts: India 8,919, US 13,345.

| Candidate tier | Cap/S1 | Link candidate recall | Mean recall per linked S1 | Avg candidates/S1 | Reduction ratio | Candidate count p95 |
|---|---:|---:|---:|---:|---:|---:|
| exact name | 50 | 20.9883% | 21.0745% | 6.05 | 99.999941% | 50 |
| exact name | 100 | 21.5336% | 21.6199% | 7.90 | 99.999923% | 58 |
| exact name | 200 | 21.8017% | 21.8982% | 9.24 | 99.999911% | 58 |
| exact name | 500 | 21.8382% | 21.9310% | 9.55 | 99.999908% | 58 |
| exact name | 1000 | 21.8382% | 21.9310% | 9.55 | 99.999908% | 58 |
| name only | 50 | 39.1499% | 39.1351% | 26.61 | 99.999742% | 50 |
| name only | 100 | 44.4663% | 44.4239% | 45.89 | 99.999555% | 100 |
| name only | 200 | 49.0513% | 48.9600% | 72.05 | 99.999302% | 200 |
| name only | 500 | 51.9417% | 51.7642% | 108.22 | 99.998951% | 500 |
| name only | 1000 | 52.6771% | 52.4803% | 127.06 | 99.998769% | 640 |
| full multistage | 50 | 61.7702% | 61.5614% | 39.63 | 99.999616% | 50 |
| full multistage | 100 | 67.3104% | 67.0437% | 73.65 | 99.999286% | 100 |
| full multistage | 200 | 72.9392% | 72.7283% | 129.50 | 99.998745% | 200 |
| full multistage | 500 | 77.4552% | 77.1874% | 226.02 | 99.997810% | 500 |
| full multistage | 1000 | 78.2334% | 77.9732% | 269.57 | 99.997388% | 878 |

At the largest evaluated full multistage cap of 1000, average candidates are 269.57 per validation S1 (p99 1,000; maximum 1,000). Candidate recall is measured over ground-truth links; singleton rows have no positive links and are excluded from the per-linked-S1 recall mean.

## Reproduction and outputs

Run `py blocking.py` from the repository root. The script rebuilds the temporary SQLite postings index from the supplied training Source 2/3 data, evaluates the deterministic held-out Source 1 bucket, removes its temporary index, and writes [`analysis/blocking_validation.json`](./analysis/blocking_validation.json) plus this report. Use `--bucket` and `--batch-size` to select another deterministic bucket or adjust query batching. No external lookup or data is used.

This is a candidate-generation baseline, not a matching model. The score used to order candidates must not be treated as a final match probability or threshold.
