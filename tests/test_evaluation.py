import numpy as np

from src.evaluation import evaluate_model


def test_evaluate_model():

    y_true = np.array([
        "Machine Learning",
        "Machine Learning",
        "Data Science",
        "Data Science"
    ])

    y_pred = np.array([
        "Machine Learning",
        "Machine Learning",
        "Data Science",
        "Data Science"
    ])

    result = evaluate_model(y_true, y_pred)

    assert result["accuracy"] == 100.0
    assert result["precision"] == 100.0
    assert result["recall"] == 100.0
    assert result["f1_score"] == 100.0

    assert result["confusion_matrix"] is not None