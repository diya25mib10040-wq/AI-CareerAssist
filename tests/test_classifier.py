import numpy as np
from scipy.sparse import csr_matrix

from src.classifier import train_career_model, predict_career


def test_career_classification():

    X_train = csr_matrix([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1]
    ])

    y_train = np.array([
        "Machine Learning",
        "Machine Learning",
        "Web Development",
        "Web Development",
        "Data Science",
        "Data Science"
    ])

    model = train_career_model(X_train, y_train)

    predictions = predict_career(
        model,
        X_train[0],
        top_n=3
    )

    assert len(predictions) == 3

    assert "Career" in predictions[0]
    assert "Confidence" in predictions[0]

    assert 0 <= predictions[0]["Confidence"] <= 100