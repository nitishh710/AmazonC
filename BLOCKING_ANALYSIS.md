# Blocking Strategy and Validation

## Scope and validation design

This stage implements candidate generation only; it does not train a matching model or predict final matches. It uses only the supplied training TSVs.

Validation source-1 entities are the deterministic 1-in-100 BLAKE2b hash bucket `17` (22,264 of 2,206,821). The remaining source-1 rows are the conceptual training partition. All training Source 2 and Source 3 rows are retained as the candidate corpus because those records are the search universe, and source-1 rows are not candidates.

## Multi-stage blocking

1. **Exact normalized name:** lowercase alphanumeric normalization; only values with at most 500 target postings are indexed.
2. **Name token overlap:** shared tokens of at least three characters, retaining only corpus document frequencies up to 3,000.
3. **Name character trigrams:** shared trigrams from names, capped at 1,000 target postings per key to tolerate small spelling edits without unbounded expansions.
4. **Address evidence:** exact normalized address (up to 1,500 postings), shared 5–6 digit address number (up to 5,000 postings), or address token of at least four characters (up to 2,000 postings).

All block keys are scoped by the record's exact country label. This uses country equality to reduce cross-country collisions without hard-coding a country list, so unseen labels remain eligible. A deterministic evidence score ranks candidates using twice the name-evidence score plus address-evidence score; exact and rare-token name evidence, character trigrams, exact address, postal-like numbers, and address tokens contribute. Scores are only for candidate ordering, not match decisions. For evaluation, each tier is independently capped at 100/250/500/1,000/2,000 candidates per S1.

## Held-out blocking results

The validation set contains 76,838 target links and 1,231 singletons (5.53%). Country counts: India 8,919, US 13,345.
The index found 76,838 of the validation target IDs in the candidate corpus; 0 of those labels cross a country boundary.

| Candidate tier | Cap/S1 | Link candidate recall | Mean recall per linked S1 | Avg candidates/S1 | Reduction ratio | Candidate count p95 |
|---|---:|---:|---:|---:|---:|---:|
| exact name | 100 | 21.5336% | 21.6199% | 7.89 | 99.999924% | 58 |
| exact name | 250 | 21.8212% | 21.9154% | 9.41 | 99.999909% | 58 |
| exact name | 500 | 21.8382% | 21.9310% | 9.52 | 99.999908% | 58 |
| exact name | 1000 | 21.8382% | 21.9310% | 9.52 | 99.999908% | 58 |
| exact name | 2000 | 21.8382% | 21.9310% | 9.52 | 99.999908% | 58 |
| name only | 100 | 50.3149% | 50.1577% | 68.18 | 99.999339% | 100 |
| name only | 250 | 55.0457% | 54.8555% | 151.62 | 99.998531% | 250 |
| name only | 500 | 57.9479% | 57.7234% | 272.52 | 99.997359% | 500 |
| name only | 1000 | 60.6080% | 60.3626% | 475.39 | 99.995394% | 1,000 |
| name only | 2000 | 63.4022% | 63.2190% | 750.58 | 99.992727% | 2,000 |
| full multistage | 100 | 77.6504% | 77.4011% | 93.99 | 99.999089% | 100 |
| full multistage | 250 | 82.1482% | 81.8500% | 226.54 | 99.997805% | 250 |
| full multistage | 500 | 84.8447% | 84.5126% | 430.74 | 99.995826% | 500 |
| full multistage | 1000 | 87.2121% | 86.9200% | 785.46 | 99.992389% | 1,000 |
| full multistage | 2000 | 90.3056% | 90.1067% | 1304.22 | 99.987362% | 2,000 |

At the largest evaluated full multistage cap of 2000, average candidates are 1304.22 per validation S1 (p99 2,000; maximum 2,000). Candidate recall is measured over ground-truth links; singleton rows have no positive links and are excluded from the per-linked-S1 recall mean.

## Reproduction and outputs

Run `py blocking.py` from the repository root. The script rebuilds the temporary SQLite postings index from the supplied training Source 2/3 data, evaluates the deterministic held-out Source 1 bucket, removes its temporary index, and writes [`analysis/blocking_validation.json`](./analysis/blocking_validation.json) plus this report. Use `--bucket` and `--batch-size` to select another deterministic bucket or adjust query batching. No external lookup or data is used.

This is a candidate-generation baseline, not a matching model. The score used to order candidates must not be treated as a final match probability or threshold.
