from sklearn.linear_model import LogisticRegression


def train_career_model(X_train, y_train):
    """
    Train a Logistic Regression model for career classification.
    """

    career_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    career_model.fit(X_train, y_train)

    return career_model


def predict_career(career_model, resume_vector, top_n=3):
    """
    Predict the top career categories for a resume.
    """

    probabilities = career_model.predict_proba(resume_vector)[0]

    top_indices = probabilities.argsort()[-top_n:][::-1]

    predictions = []

    for index in top_indices:
        predictions.append({
            "Career": career_model.classes_[index],
            "Confidence": round(probabilities[index] * 100, 2)
        })

    return predictions