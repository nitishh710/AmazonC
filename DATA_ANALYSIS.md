# Dataset Profiling and EDA

## Scope and reproducibility

This analysis uses only the TSV files supplied in `dataset/`. It describes the
provided records and training labels; it does not perform blocking, fit or
evaluate a matching model, or use external entity data.

Reproduce the profile from the repository root with:

```powershell
py profile_dataset.py
```

The streaming profiler writes the detailed counts and histograms to
[`analysis/profiling_summary.json`](./analysis/profiling_summary.json). It
validates the source and ground-truth headers and processes one source at a
time. Text duplicate patterns use lowercase alphanumeric normalization
(non-alphanumeric runs become spaces).

## Dataset scale and identifier integrity

| Split | Source 1 rows | Source 2 rows | Source 3 rows | Total source records |
|---|---:|---:|---:|---:|
| Train | 2,206,821 | 5,034,616 | 5,285,603 | 12,527,040 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 | 11,702,133 |

All six source files have the expected four columns (`entity_id`,
`business_name`, `business_address`, `country`). Every source-file entity ID is
unique; no duplicate IDs were found. The training ground truth has exactly one
row per training Source 1 record.

## Country distribution and test-time shift

| Split/source | US | India | France |
|---|---:|---:|---:|
| Train S1 | 1,323,633 | 883,188 | 0 |
| Train S2 | 3,016,817 | 2,017,799 | 0 |
| Train S3 | 3,170,056 | 2,115,547 | 0 |
| Test S1 | 663,106 | 809,986 | 259,452 |
| Test S2 | 1,871,330 | 2,312,565 | 703,378 |
| Test S3 | 1,945,701 | 2,405,000 | 731,615 |

France is approximately 14.4–15.0% of each test source, despite being absent
from training. Country must therefore be treated as an open-set string
attribute, not as a closed US/India enumeration. Country equality can be useful
for comparisons, but should not cause unseen-country records to be discarded.

## Missing values and text characteristics

The profiler treats blank strings and the tokens `na`, `n/a`, `null`, `none`,
`nan`, and `-` as missing when counting field completeness.

| Source | Missing names | Missing addresses |
|---|---:|---:|
| Train S1 | 0 | 0 |
| Train S2 | 6 | 168,967 (3.36%) |
| Train S3 | 18 | 175,916 (3.33%) |
| Test S1 | 0 | 0 |
| Test S2 | 49 | 129,408 (2.65%) |
| Test S3 | 61 | 136,098 (2.68%) |

Missing addresses are concentrated in Sources 2 and 3. Name fields are
effectively complete. Address-based features and blocking must tolerate
missingness rather than assume an address is present.

Names most commonly have lengths of 20–39 characters. Addresses are generally
longer: the modal length bin is 20–39 characters for S1 and 40–79 characters
for S2/S3. Detailed per-source length histograms are in the JSON summary.
Punctuation appears in about 21% of Source 1 names and 40–43% of Source 2/3
names. More than 90% of addresses contain digits, and all S1 addresses contain
commas in these files. The profiler's “postal-like number” count is only a
coarse 5–6 digit regex, not a validated postal-code extraction.

These patterns support using normalized and token/character-based name and
address comparisons in later work, while retaining features that can still
operate when an address is absent. Simple normalization is only a profiling
device here; it does not resolve abbreviations, transliteration, token order,
or address semantics.

## Repeated text and record patterns

Normalized name values repeat frequently: rows beyond the first in duplicate
name groups range from about 0.50 million in test S1 to 1.21 million in train
S2. Repeated normalized addresses are also common, including groups containing
missing addresses. These counts are not entity-match counts: common business
names, generic addresses, and values that normalize to the same text can refer
to different businesses. In particular, a large duplicate group must not be
used as a match decision by itself.

There are no repeated normalized `(business_name, business_address, country)`
records in train S1. Exact normalized non-ID record duplicates do occur in the
other source files:

| Source | Rows in duplicate-record groups | Rows beyond first in each group |
|---|---:|---:|
| Train S2 | 141,434 | 72,756 |
| Train S3 | 102,294 | 52,328 |
| Test S1 | 2 | 1 |
| Test S2 | 131,029 | 67,454 |
| Test S3 | 88,900 | 45,463 |

The grouping uses normalized name, address, and country, so it can collapse
punctuation/case differences; it should not be interpreted as literal
byte-for-byte duplication or as proof of entity identity.

## Training-label structure

| Ground-truth measure | Count |
|---|---:|
| Source 1 rows with a ground-truth row | 2,206,821 |
| Total Source 1-to-target links | 7,638,365 |
| Mean links per Source 1 record | 3.46 |
| Source 1 records with no links (singletons) | 123,247 (5.58%) |
| Links to Source 2 | 3,693,619 |
| Links to Source 3 | 3,944,746 |

| Ground-truth target sources present for an S1 | S1 count | Share |
|---|---:|---:|
| Both S2 and S3 | 1,776,047 | 80.48% |
| S2 only | 143,029 | 6.48% |
| S3 only | 164,498 | 7.45% |
| Neither (singleton) | 123,247 | 5.58% |

The number of linked target IDs ranges from zero to eleven; the full
distribution is in the JSON summary. Each target ID occurs at most once in the
ground-truth mappings. The substantial one-to-many label structure means that
future evaluation should assess all target links for each S1 record rather
than assume a single match.

## Implications for subsequent stages

- Keep country as an open-set comparison feature so the test-only France
  records remain eligible.
- Design candidate generation and matching to support multiple S2/S3 matches
  per S1 and explicit no-match outcomes.
- Avoid relying on exact normalized name or address equality alone: repeated
  values are common, and missing addresses are concentrated in S2/S3.
- Report candidate recall and candidate-set size separately in later blocking
  experiments; this EDA does not measure either.

No candidate-generation logic, train/validation split, blocking, matching
model, or test prediction has been implemented as part of this profiling task.
