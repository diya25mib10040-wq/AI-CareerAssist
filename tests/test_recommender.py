from src.recommender import generate_learning_recommendations


def test_learning_recommendations():

    missing_skills = [
        "Python",
        "Machine Learning",
        "SQL"
    ]

    recommendations = generate_learning_recommendations(
        missing_skills
    )

    assert len(recommendations) == 3

    assert recommendations[0] == "Learn Python programming"
    assert recommendations[1] == "Learn Machine Learning fundamentals"
    assert recommendations[2] == "Learn SQL and database management"


def test_unknown_skill_recommendation():

    missing_skills = ["TensorFlow"]

    recommendations = generate_learning_recommendations(
        missing_skills
    )

    assert len(recommendations) == 1
    assert recommendations[0] == "Learn and practice TensorFlow"