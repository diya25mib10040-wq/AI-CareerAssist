import pandas as pd


def convert_skills_to_list(skill_text):
    """
    Convert skill text into a list of individual skills.
    """

    if pd.isna(skill_text):
        return []

    skill_text = str(skill_text).lower()

    skill_text = skill_text.replace("|", ",")
    skill_text = skill_text.replace(";", ",")
    skill_text = skill_text.replace("\n", ",")

    skills = [skill.strip() for skill in skill_text.split(",")]

    skills = [skill for skill in skills if skill]

    return skills


def analyze_skill_gap(resume_skills, job_skills):
    """
    Compare resume skills with job-required skills.
    """

    resume_skills = set(
        skill.lower().strip()
        for skill in resume_skills
    )

    job_skills = set(
        skill.lower().strip()
        for skill in job_skills
    )

    matched_skills = resume_skills.intersection(job_skills)

    missing_skills = job_skills.difference(resume_skills)

    return {
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills),
        "total_required_skills": len(job_skills),
        "total_matched_skills": len(matched_skills),
        "total_missing_skills": len(missing_skills)
    }