from research_lab.evaluation import sequence_metrics


def test_sequence_metrics_counts_missing_and_extra_tokens():
    metrics = sequence_metrics([["a", "b"], ["a", "extra"]], [["a", "b"], ["a"]])
    assert metrics["exact_match"] == 0.5
    assert metrics["token_accuracy"] == 3 / 4
