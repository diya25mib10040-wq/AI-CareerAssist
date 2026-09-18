from src.skill_gap import convert_skills_to_list, analyze_skill_gap


def test_convert_skills_to_list():
    skills = convert_skills_to_list("Python, Machine Learning, SQL")

    assert "python" in skills
    assert "machine learning" in skills
    assert "sql" in skills


def test_skill_gap_analysis():
    resume_skills = ["Python", "SQL", "Machine Learning"]
    job_skills = ["Python", "SQL", "Deep Learning"]

    result = analyze_skill_gap(resume_skills, job_skills)

    assert "python" in result["matched_skills"]
    assert "sql" in result["matched_skills"]
    assert "deep learning" in result["missing_skills"]