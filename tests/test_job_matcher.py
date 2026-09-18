import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.job_matcher import find_best_job


def test_find_best_job():
    job_df = pd.DataFrame({
        "Title": [
            "Machine Learning Engineer",
            "Web Developer"
        ],
        "ExperienceLevel": [
            "Entry Level",
            "Entry Level"
        ],
        "YearsOfExperience": [
            1,
            1
        ],
        "Skills": [
            "Python, Machine Learning",
            "HTML, CSS, JavaScript"
        ]
    })

    vectorizer = TfidfVectorizer()

    job_text = [
        "Python Machine Learning",
        "HTML CSS JavaScript"
    ]

    job_vectors = vectorizer.fit_transform(job_text)

    resume_vector = vectorizer.transform(
        ["Python Machine Learning"]
    )

    result = find_best_job(
        resume_vector,
        job_vectors,
        job_df
    )

    assert result["Job Title"] == "Machine Learning Engineer"
    assert result["Experience Level"] == "Entry Level"
    assert result["Match Score"] > 0