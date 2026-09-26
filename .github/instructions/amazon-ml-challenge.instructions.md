---
description: Guidelines for the Amazon ML Challenge business entity resolution project
applyTo: "**"
---
<!-- Tip: Use /create-instructions in chat to generate content with agent assistance -->

Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.
# Amazon ML Challenge — Business Entity Resolution

You are working on the Amazon ML Challenge 2026 Business Entity Resolution project.

## Project rules
- Read PROJECT_BRIEF.md before making major architectural decisions.
- Read DATA_ANALYSIS.md before doing EDA or modifying the data pipeline.
- Do NOT repeat completed EDA unless there is a specific reason.
- Do NOT use external data, web lookups, business registries, geocoding APIs, commercial entity-resolution APIs, or any external augmentation.
- All processing must use the provided train/test datasets.

## Objective
Match every Source 1 entity to zero, one, or multiple matching entities from Source 2 and Source 3.

Optimize for macro F0.5. Precision is important because false matches are costly.

## Development rules
- Inspect the existing code before creating new files.
- Reuse working components instead of rewriting them.
- Work incrementally and keep stages modular.
- Do not implement a later stage while an earlier requested stage is still being evaluated.
- Do not silently change the project architecture.
- Use reproducible random seeds.
- Prefer efficient pandas/numpy/sklearn implementations.
- Avoid unnecessary copies of the large datasets.
- Log important metrics and intermediate results.

## Candidate generation
- Candidate generation/blocking is critical.
- Candidate recall must be measured against training ground truth.
- Track candidate recall, average candidates per Source 1, candidate reduction ratio, and coverage.
- candidate_pairs.tsv must contain the FINAL candidate set actually passed to the final matching model.
- Final predicted matches must always be a subset of candidate_pairs.tsv.

## Matching
- Evaluate precision/recall/F0.5 on a validation split before finalizing thresholds.
- Prefer precision-safe matching decisions.
- Treat singleton and empty-match cases explicitly.
- Do not assume every Source 1 entity has a match.

## Validation
Before considering a stage complete:
1. Run relevant tests.
2. Check generated files and row counts.
3. Run the official submission validator when output files exist.
4. Report what changed and the measured metrics.

## Communication
Before coding:
- Briefly state what you inspected and what you plan to change.

After coding:
- Summarize files changed.
- Report tests/validation performed.
- Report important metrics.
- Clearly mention any remaining uncertainty or limitation.

Do not claim an improvement unless it is supported by measured validation results.