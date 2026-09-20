"""Greek sentiment analysis extension.

Fine-tunes nlpaueb/bert-base-greek-uncased-v1 on DGurgurov/greek_sa
(Tsakalidis et al. 2018) with the same provenance discipline as the
English pipeline: pinned dataset revision, manifest, saved test logits
in the unified-benchmark format.

Everything is driven by configs/greek_bert.yaml, parsed once into the frozen
dataclasses in `config.py`. That module is the single place a setting is
declared, and it rejects keys it does not honour, so the file and the code
cannot drift apart unnoticed.

Module map:
    config.py    typed sections; the config/code contract
    data.py      download, clean, group duplicates, split, check leakage
    baseline.py  TF-IDF + logistic regression floor on the same frames
    model.py     tokenizer and classification head
    train.py     Trainer loop with early stopping, writes metrics + logits
    evaluate.py  score a saved checkpoint on the configured splits
    predict.py   inference helpers

Deliberately imports nothing heavy at package level so that `import
reviewnlp` and the CI test run never need torch or a model download.
"""
