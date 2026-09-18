from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_features(resume_texts, job_texts):
    """
    Convert resume and job description text into
    TF-IDF numerical features.
    """

    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    # Learn vocabulary from resume text
    resume_tfidf = tfidf_vectorizer.fit_transform(resume_texts)

    # Convert job text using the same vocabulary
    job_tfidf = tfidf_vectorizer.transform(job_texts)

    return tfidf_vectorizer, resume_tfidf, job_tfidf