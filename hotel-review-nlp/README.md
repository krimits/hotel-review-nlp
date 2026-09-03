# Hotel Review NLP Project - Week 1 Progress

## Dataset Methodology
The dataset splits and distribution after removing ambiguous reviews:
* **Total Labeled Reviews:** 147,140
* **Train Set:** 118,990 rows (92,758 positive / 26,232 negative)
* **Dev Set:** 14,872 rows (11,594 positive / 3,278 negative)
* **Test Set:** 13,278 rows (10,000 positive / 3,278 negative)

## Baseline Evaluation Results
Below are the performance metrics for the classical machine learning models:

| Model | Macro-F1 | Accuracy |
| :--- | :---: | :---: |
| **naive_bayes_word** | **0.9345** | **95.09%%** |
| lr_sgd_word | 0.9173 | 94.11%% |
| naive_bayes_char | 0.8545 | 89.91%% |
| lr_sgd_char | 0.5681 | 78.87%% |

The best performing model is **naive_bayes_word** with a Macro-F1 of **0.9345**. The trained artifact has been saved to `runs/classical/best_classical.joblib`.
