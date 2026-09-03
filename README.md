# Hotel Review NLP Project - Week 1 Progress

## Dataset Methodology
The dataset splits and distribution after removing ambiguous reviews:
* **Total Labeled Reviews:** 147,140
* **Train Set:** 118,990 rows (92,758 positive / 26,232 negative)
* **Dev Set:** 14,872 rows (11,594 positive / 3,278 negative)
* **Test Set:** 13,278 rows (10,000 positive / 3,278 negative)

## Evaluation Results (Week 1)
Below are the benchmark metrics comparing classical Machine Learning with Deep Learning:

| Model Architecture | Macro-F1 | Accuracy | Status |
| :--- | :---: | :---: | :--- |
| **BiLSTM (Pure PyTorch)** | **0.9503** | **96.29%%** | ?? **Current Best** |
| naive_bayes_word | 0.9345 | 95.09%% | Best Classical Baseline |
| lr_sgd_word | 0.9173 | 94.11%% | Classical Baseline |
| naive_bayes_char | 0.8545 | 89.91%% | Classical Baseline |
| lr_sgd_char | 0.5681 | 78.87%% | Failed Baseline |

The Deep Learning model **BiLSTM** achieved the highest Macro-F1 of **0.9503** on the test set, demonstrating the value of sequential text processing over simple bag-of-words counting.
