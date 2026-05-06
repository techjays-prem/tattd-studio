# Anatomy Critic — Eval Report

- Run: 2026-05-06T11:42:08.300311+00:00
- Golden Set: `data/eval/anatomy_cases.jsonl` (60 cases)
- Judge: `heuristic-baseline-v1`

| Metric | Value | Threshold | Status |
|---|---|---|---|
| Precision | 0.967 | 0.70 | PASS |
| Recall | 0.967 | 0.85 | PASS |

## Confusion matrix

Positive class: invalid placement (the verdict the Critic must catch).

| | Predicted invalid | Predicted valid |
|---|---|---|
| **Actually invalid** | TP=29 | FN=1 |
| **Actually valid** | FP=1 | TN=29 |
