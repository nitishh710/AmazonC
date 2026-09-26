# AMAZON ML CHALLENGE 2026 — PROJECT BRIEF

## OBJECTIVE

Solve the Amazon ML Challenge 2026 Business Entity Resolution problem.

We have 3 independent sources:

- Source 1 = deduplicated reference entities
- Source 2
- Source 3

For EVERY Source 1 entity, identify all matching Source 2/Source 3 records referring to the same real-world business.

Training:
- train_source1.tsv
- train_source2.tsv
- train_source3.tsv
- train_ground_truth.tsv

Test:
- test_source1.tsv
- test_source2.tsv
- test_source3.tsv

All files are TSV. Always use sep="\t".

## OUTPUTS

Must generate:

output/matching_results.tsv
output/candidate_pairs.tsv

matching_results.tsv:
source1_entity_id    matched_entity_ids

candidate_pairs.tsv:
source1_entity_id    candidate_entity_ids

Every final match MUST be present in candidate_pairs.tsv.

Every test Source 1 entity MUST appear exactly once.

Empty match list is valid.

## EVALUATION

Metric = macro F0.5.

F0.5 prioritizes precision over recall.

False merges are particularly costly.

Singleton detection is important:
- predicting no match when there is no true match = score 1
- predicting an incorrect match for a singleton = score 0

## MOST IMPORTANT REQUIREMENT: BLOCKING

Amazon explicitly evaluates candidate generation.

We need:
1. High candidate recall
2. Very small candidate sets

candidate_pairs.tsv must contain the FINAL candidate set actually passed to the matching model.

Do NOT generate huge all-pairs candidates.

Measure:
- candidate recall
- average candidates per Source 1
- reduction ratio
- validation F0.5

## EXPECTED NOISE

Business names:
- typos
- abbreviations
- legal suffixes
- Corp/Corporation
- Pvt/Private
- Ltd/Limited
- punctuation
- token order changes
- transliteration
- DBA/trade names

Addresses:
- Rd/Road
- St/Street
- missing components
- reordered components
- landmarks
- transliteration
- municipal numbering variations
- missing PIN/state/etc.

Country is an OPEN SET.
Do NOT hard-code US/India.
Test contains France.

## RECOMMENDED PIPELINE

1. Data exploration
2. Train/validation split
3. Normalization
4. Blocking / candidate generation
5. Pairwise feature engineering
6. Matching model
7. Threshold optimization for macro F0.5
8. Singleton handling
9. Generate candidate_pairs.tsv
10. Generate matching_results.tsv
11. Validate submission
12. Error analysis
13. Iterate

## POSSIBLE FEATURES

Name:
- exact normalized match
- character n-gram TF-IDF cosine
- token Jaccard
- token overlap
- edit similarity
- token-set similarity
- length differences
- rare-token overlap

Address:
- character similarity
- token Jaccard
- TF-IDF cosine
- postcode/PIN overlap
- component overlap
- length differences

Other:
- country equality
- source pair (S1-S2 / S1-S3)
- name/address combined features

## MODEL

Start with strong classical ML baselines.

Candidates to evaluate:
- Logistic Regression
- LightGBM
- XGBoost
- CatBoost

Prefer models with MIT/Apache 2.0 compatible licensing and <=8B parameters.

Do NOT introduce an external pretrained model without checking its license.

## VALIDATION

Create a validation split from training data.

Evaluate:
- macro F0.5
- precision
- recall
- singleton accuracy
- candidate recall
- average candidates per S1

Optimize decision threshold using validation data.

Do NOT simply use probability > 0.5.

## STRICT CHALLENGE RULE

External entity lookup is PROHIBITED.

Do NOT use:
- Google/business search
- business databases
- government registries
- geocoding APIs
- external entity-resolution APIs
- internet-based business information
- external data augmentation

Use ONLY the supplied challenge data.

## CODING PRINCIPLES

- Work directly in this repository.
- Inspect existing files before creating new architecture.
- Keep code modular and reproducible.
- Avoid unnecessary dependencies.
- Do not delete useful existing files.
- Save experiments/results clearly.
- Do not fabricate validation scores.
- Document important decisions.
- Run the official validator before submission.

## SUCCESS CRITERIA

The final solution should maximize validation macro F0.5 while maintaining:
- high blocking recall
- low candidate count
- low false-positive rate
- correct singleton handling
- valid submission format

Before declaring completion, verify the entire pipeline end-to-end on the test data.