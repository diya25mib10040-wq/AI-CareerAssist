learning_resources = {
    "python": "Learn Python programming",
    "machine learning": "Learn Machine Learning fundamentals",
    "deep learning": "Learn Deep Learning and neural networks",
    "sql": "Learn SQL and database management",
    "data analysis": "Learn Data Analysis with Pandas and NumPy",
    "excel": "Improve Excel and data handling skills",
    "communication": "Improve communication and presentation skills",
    "project management": "Learn project management fundamentals",
    "data visualization": "Learn data visualization using Matplotlib and Seaborn",
    "statistics": "Learn statistics for data analysis",
    "html": "Learn HTML and web page structure",
    "css": "Learn CSS and web styling",
    "javascript": "Learn JavaScript programming",
    "java": "Learn Java programming",
    "c++": "Learn C++ programming",
    "git": "Learn Git and version control",
    "github": "Learn GitHub and collaborative development",
    "cloud": "Learn cloud computing fundamentals",
    "aws": "Learn AWS cloud services",
    "docker": "Learn Docker and containerization"
}


def generate_learning_recommendations(missing_skills):
    """
    Generate learning recommendations based on missing skills.
    """

    recommendations = []

    for skill in missing_skills:

        skill_lower = skill.lower().strip()

        if skill_lower in learning_resources:
            recommendations.append(
                learning_resources[skill_lower]
            )
        else:
            recommendations.append(
                f"Learn and practice {skill}"
            )

    return recommendations