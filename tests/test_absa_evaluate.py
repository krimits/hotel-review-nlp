from __future__ import annotations

import pytest

from reviewnlp.absa.evaluate import evaluate_annotations


def test_aspect_metrics_use_human_labels_and_review_identity():
    gold = [
        {"row_id": 1, "text": "dirty room, kind staff", "status": "annotated", "aspects": [
            {"aspect": "cleanliness", "sentiment": "negative", "quote": "dirty room"},
            {"aspect": "staff", "sentiment": "positive", "quote": "kind staff"}]},
        {"row_id": 2, "text": "Great pool", "status": "annotated", "aspects": [
            {"aspect": "facilities", "sentiment": "positive", "quote": "Great pool"}]},
    ]
    predictions = [
        {"row_id": 2, "text": "Great pool", "aspects": []},
        {"row_id": 1, "text": "dirty room, kind staff", "aspects": [
            {"aspect": "cleanliness", "sentiment": "negative", "quote": "dirty room"},
            {"aspect": "staff", "sentiment": "negative", "quote": "kind staff"},
            {"aspect": "food", "sentiment": "positive", "quote": "hallucinated"}]},
    ]
    result = evaluate_annotations(gold, predictions)
    assert result["aspect_micro_f1"] == 0.6667  # TP=2, FP=1, FN=1
    assert result["matched_sentiment_accuracy"] == 0.5
    assert result["invalid_predicted_quotes"] == 1


def test_unreviewed_gold_and_misaligned_predictions_cannot_be_scored():
    gold = [{"row_id": 1, "text": "kind staff", "status": "unlabeled", "aspects": []}]
    pred = [{"row_id": 1, "text": "kind staff", "aspects": []}]
    with pytest.raises(ValueError, match="not been completed"):
        evaluate_annotations(gold, pred)
    gold[0]["status"] = "annotated"
    pred[0]["text"] = "dirty room"
    with pytest.raises(ValueError, match="text differs"):
        evaluate_annotations(gold, pred)
