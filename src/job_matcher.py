from sklearn.metrics.pairwise import cosine_similarity


def find_best_job(resume_vector, job_tfidf, job_df):
    """
    Find the job that is most similar to the resume.
    """

    similarity_scores = cosine_similarity(
        resume_vector,
        job_tfidf
    )[0]

    best_job_index = similarity_scores.argmax()
    best_score = similarity_scores[best_job_index]

    best_job = job_df.iloc[best_job_index]

    return {
        "Job Title": best_job["Title"],
        "Experience Level": best_job["ExperienceLevel"],
        "Years of Experience": best_job["YearsOfExperience"],
        "Skills": best_job["Skills"],
        "Match Score": round(best_score * 100, 2)
    }