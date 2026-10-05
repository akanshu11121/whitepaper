"""Sequence-specific free-running metrics; teacher-forced NLL is tracked separately."""


def sequence_metrics(predictions: list[list[str]], targets: list[list[str]]) -> dict[str, float]:
    if not targets or len(predictions) != len(targets):
        raise ValueError("aligned nonempty predictions and targets required")
    exact = sum(pred == target for pred, target in zip(predictions, targets, strict=True))
    correct = sum(sum(a == b for a, b in zip(pred, target, strict=False)) for pred, target in zip(predictions, targets, strict=True))
    # Extra generated tokens are errors, as are omissions.
    count = sum(max(len(pred), len(target)) for pred, target in zip(predictions, targets, strict=True))
    return {"exact_match": exact / len(targets), "token_accuracy": correct / max(count, 1)}
