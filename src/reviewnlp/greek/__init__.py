"""Greek sentiment analysis extension.

Fine-tunes nlpaueb/bert-base-greek-uncased-v1 on DGurgurov/greek_sa
(Tsakalidis et al. 2018) with the same provenance discipline as the
English pipeline: pinned dataset revision, manifest, saved test logits
in the unified-benchmark format.

Deliberately imports nothing heavy at package level so that `import
reviewnlp` and the CI test run never need torch or a model download.
"""
