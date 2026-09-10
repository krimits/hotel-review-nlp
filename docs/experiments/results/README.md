# Preserved legacy metrics

These JSON files are immutable copies of the metrics that existed before the repository was flattened on 2026-09-10.

- `classical_legacy_metrics.json` records the original four classical runs. The entries named `*_char` were produced before the missing `analyzer="char"` bug was fixed and therefore must not be treated as real character n-gram results.
- `bilstm_legacy_metrics.json` records the seed-42 BiLSTM result on the frozen legacy test split.

New final results belong in `runs/benchmark/` and must include predictions plus the common test-set fingerprint.
