import streamlit as st
import pandas as pd
import re
from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI CareerAssist",
    page_icon="💼",
    layout="wide"
)

st.title("🤖 AI CareerAssist")

st.subheader(
    "AI-Powered Resume Analysis, Job Matching and Skill Gap Recommendation System"
)

st.write(
    "Upload your resume to get job matching, skill gap analysis, "
    "career recommendations, and learning recommendations."
)

st.divider()


# =========================================================
# LOAD DATASETS
# =========================================================

@st.cache_data
def load_data():

    # Find project folder
    project_folder = Path(__file__).resolve().parent.parent

    # Find data folder
    data_folder = project_folder / "data"

    resume_file = data_folder / "Resume.csv"
    job_file = data_folder / "job_dataset.csv"

    # Check files exist
    if not resume_file.exists():
        st.error(
            f"Resume.csv was not found at: {resume_file}"
        )
        st.stop()

    if not job_file.exists():
        st.error(
            f"job_dataset.csv was not found at: {job_file}"
        )
        st.stop()

    # Load datasets
    resume_df = pd.read_csv(resume_file)
    job_df = pd.read_csv(job_file)

    return resume_df, job_df


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# FIND RESUME TEXT COLUMN
# =========================================================

def find_resume_text_column(resume_df):

    columns = resume_df.columns.tolist()

    # First try common names
    possible_names = [
        "resume_text",
        "Resume_str",
        "resume",
        "text",
        "Resume"
    ]

    for name in possible_names:

        for column in columns:

            if column.lower().strip() == name.lower():

                return column

    # Look for columns containing useful words
    for column in columns:

        column_lower = column.lower()

        if (
            "resume" in column_lower
            and "text" in column_lower
        ):

            return column

    # Look for a long text/object column
    object_columns = resume_df.select_dtypes(
        include=["object"]
    ).columns

    best_column = None
    best_length = 0

    for column in object_columns:

        column_lower = column.lower()

        # Don't use category or skill columns
        if (
            "categor" in column_lower
            or "skill" in column_lower
            or "id" in column_lower
        ):
            continue

        try:

            average_length = (
                resume_df[column]
                .fillna("")
                .astype(str)
                .str.len()
                .mean()
            )

            if average_length > best_length:

                best_length = average_length
                best_column = column

        except Exception:
            pass

    return best_column


# =========================================================
# FIND CATEGORY COLUMN
# =========================================================

def find_category_column(resume_df):

    columns = resume_df.columns.tolist()

    # Look for category column
    for column in columns:

        column_lower = column.lower()

        if (
            "categor" in column_lower
            or "category" in column_lower
        ):

            return column

    return None


# =========================================================
# READ PDF
# =========================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + " "

    return text


# =========================================================
# READ TXT
# =========================================================

def extract_txt_text(uploaded_file):

    return uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )


# =========================================================
# EXTRACT SKILLS
# =========================================================

def extract_skills(resume_text, job_df):

    resume_text = resume_text.lower()

    all_skills = []

    for skill_text in job_df["Skills"].dropna():

        skill_text = str(skill_text)

        skill_text = skill_text.replace(
            "|",
            ","
        )

        skill_text = skill_text.replace(
            ";",
            ","
        )

        skills = skill_text.split(",")

        for skill in skills:

            skill = skill.strip().lower()

            if (
                skill
                and skill not in all_skills
            ):

                all_skills.append(skill)

    detected_skills = []

    for skill in all_skills:

        if skill in resume_text:

            detected_skills.append(skill)

    return sorted(detected_skills)


# =========================================================
# LEARNING RECOMMENDATIONS
# =========================================================

def generate_learning_recommendations(
    missing_skills
):

    learning_resources = {

        "python":
            "Learn Python programming",

        "machine learning":
            "Learn Machine Learning fundamentals",

        "deep learning":
            "Learn Deep Learning and neural networks",

        "sql":
            "Learn SQL and database management",

        "excel":
            "Improve Excel and data handling skills",

        "data analysis":
            "Learn Data Analysis with Pandas and NumPy",

        "communication":
            "Improve communication and presentation skills",

        "data visualization":
            "Learn data visualization using Matplotlib and Seaborn",

        "statistics":
            "Learn statistics for data analysis",

        "html":
            "Learn HTML and web page structure",

        "css":
            "Learn CSS and web styling",

        "javascript":
            "Learn JavaScript programming",

        "java":
            "Learn Java programming",

        "c++":
            "Learn C++ programming",

        "git":
            "Learn Git and version control",

        "github":
            "Learn GitHub and collaborative development",

        "cloud":
            "Learn cloud computing fundamentals",

        "aws":
            "Learn AWS cloud services",

        "docker":
            "Learn Docker and containerization"
    }

    recommendations = []

    for skill in missing_skills:

        skill = skill.lower().strip()

        if skill in learning_resources:

            recommendations.append(
                learning_resources[skill]
            )

        else:

            recommendations.append(
                f"Learn and practice {skill}"
            )

    return recommendations


# =========================================================
# RESUME UPLOAD
# =========================================================

st.header("📄 Resume Input")

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "txt"],
    key="resume_uploader"
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button("🚀 Analyze Resume"):

    if uploaded_file is None:

        st.warning(
            "Please upload a resume first."
        )

    else:

        with st.spinner(
            "Analyzing your resume..."
        ):

            # =================================================
            # LOAD DATA
            # =================================================

            resume_df, job_df = load_data()


            # =================================================
            # FIND IMPORTANT COLUMNS
            # =================================================

            resume_text_column = (
                find_resume_text_column(
                    resume_df
                )
            )

            category_column = (
                find_category_column(
                    resume_df
                )
            )


            # =================================================
            # CHECK RESUME TEXT COLUMN
            # =================================================

            if resume_text_column is None:

                st.error(
                    "I could not find the resume text column "
                    "in Resume.csv."
                )

                st.write(
                    "Columns found in Resume.csv:"
                )

                st.write(
                    resume_df.columns.tolist()
                )

                st.stop()


            # =================================================
            # CHECK CATEGORY COLUMN
            # =================================================

            if category_column is None:

                st.error(
                    "I could not find the career category "
                    "column in Resume.csv."
                )

                st.write(
                    "Columns found in Resume.csv:"
                )

                st.write(
                    resume_df.columns.tolist()
                )

                st.stop()


            # =================================================
            # EXTRACT UPLOADED RESUME TEXT
            # =================================================

            if uploaded_file.name.lower().endswith(
                ".pdf"
            ):

                resume_text = extract_pdf_text(
                    uploaded_file
                )

            else:

                resume_text = extract_txt_text(
                    uploaded_file
                )


            # =================================================
            # CHECK RESUME TEXT
            # =================================================

            if not resume_text.strip():

                st.error(
                    "Could not extract text from the resume."
                )

                st.stop()


            # =================================================
            # CLEAN UPLOADED RESUME
            # =================================================

            clean_resume = clean_text(
                resume_text
            )


            # =================================================
            # 1. RESUME ANALYSIS
            # =================================================

            st.header(
                "📊 Resume Analysis"
            )

            word_count = len(
                clean_resume.split()
            )

            detected_skills = extract_skills(
                clean_resume,
                job_df
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Resume Words",
                    word_count
                )


            with col2:

                st.metric(
                    "Detected Skills",
                    len(detected_skills)
                )


            if detected_skills:

                st.write(
                    "### 🛠️ Detected Skills"
                )

                st.write(
                    ", ".join(
                        detected_skills
                    )
                )

            else:

                st.info(
                    "No matching skills were detected."
                )


            # =================================================
            # 2. PREPARE RESUME DATA
            # =================================================

            resume_df[
                "clean_resume_text"
            ] = (
                resume_df[
                    resume_text_column
                ]
                .fillna("")
                .astype(str)
                .apply(clean_text)
            )


            # =================================================
            # PREPARE JOB DATA
            # =================================================

            required_job_columns = [
                "Responsibilities",
                "Skills",
                "Title",
                "ExperienceLevel"
            ]

            missing_job_columns = [
                column
                for column in required_job_columns
                if column not in job_df.columns
            ]

            if missing_job_columns:

                st.error(
                    "Some required columns are missing "
                    "from job_dataset.csv:"
                )

                st.write(
                    missing_job_columns
                )

                st.stop()


            job_df[
                "clean_responsibilities"
            ] = (
                job_df[
                    "Responsibilities"
                ]
                .fillna("")
                .astype(str)
                .apply(clean_text)
            )


            # =================================================
            # 3. TF-IDF FEATURE EXTRACTION
            # =================================================

            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english"
            )


            resume_vectors = (
                vectorizer.fit_transform(
                    resume_df[
                        "clean_resume_text"
                    ]
                )
            )


            job_vectors = (
                vectorizer.transform(
                    job_df[
                        "clean_responsibilities"
                    ]
                )
            )


            uploaded_vector = (
                vectorizer.transform(
                    [clean_resume]
                )
            )


            # =================================================
            # 4. JOB MATCHING
            # =================================================

            similarities = cosine_similarity(
                uploaded_vector,
                job_vectors
            )[0]


            best_job_index = (
                similarities.argmax()
            )


            best_score = (
                similarities[
                    best_job_index
                ]
            )


            best_job = job_df.iloc[
                best_job_index
            ]


            st.header(
                "🎯 Best Job Match"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Job Title",
                    str(
                        best_job["Title"]
                    )
                )


            with col2:

                st.metric(
                    "Match Score",
                    f"{best_score * 100:.2f}%"
                )


            with col3:

                st.metric(
                    "Experience Level",
                    str(
                        best_job[
                            "ExperienceLevel"
                        ]
                    )
                )


            st.write(
                "**Required Skills:**",
                best_job["Skills"]
            )


            # =================================================
            # 5. SKILL GAP ANALYSIS
            # =================================================

            job_skills_text = str(
                best_job["Skills"]
            ).lower()


            job_skills_text = (
                job_skills_text
                .replace("|", ",")
                .replace(";", ",")
            )


            required_skills = set(

                skill.strip()

                for skill
                in job_skills_text.split(",")

                if skill.strip()
            )


            resume_skills = set(
                detected_skills
            )


            matched_skills = (
                required_skills
                .intersection(
                    resume_skills
                )
            )


            missing_skills = (
                required_skills
                - resume_skills
            )


            st.header(
                "🔍 Skill Gap Analysis"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.subheader(
                    "✅ Matched Skills"
                )


                if matched_skills:

                    for skill in sorted(
                        matched_skills
                    ):

                        st.write(
                            "•",
                            skill
                        )

                else:

                    st.write(
                        "No matched skills found."
                    )


            with col2:

                st.subheader(
                    "❌ Missing Skills"
                )


                if missing_skills:

                    for skill in sorted(
                        missing_skills
                    ):

                        st.write(
                            "•",
                            skill
                        )

                else:

                    st.success(
                        "No major skill gaps found!"
                    )


            # =================================================
            # 6. CAREER CLASSIFICATION
            # =================================================

            st.header(
                "💼 Career Recommendations"
            )


            X = resume_vectors

            y = resume_df[
                category_column
            ].fillna(
                "Unknown"
            )


            # Train Logistic Regression
            career_model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )


            career_model.fit(
                X,
                y
            )


            # Predict probabilities
            probabilities = (
                career_model
                .predict_proba(
                    uploaded_vector
                )[0]
            )


            # Number of available classes
            number_of_classes = len(
                career_model.classes_
            )


            # Show maximum 3 recommendations
            number_to_show = min(
                3,
                number_of_classes
            )


            top_indices = (
                probabilities
                .argsort()[
                    -number_to_show:
                ][::-1]
            )


            for rank, index in enumerate(
                top_indices,
                start=1
            ):

                career = (
                    career_model
                    .classes_[index]
                )


                confidence = (
                    probabilities[index]
                    * 100
                )


                st.write(
                    f"**{rank}. {career}** — "
                    f"{confidence:.2f}%"
                )


            # =================================================
            # 7. LEARNING RECOMMENDATIONS
            # =================================================

            st.header(
                "📚 Learning Recommendations"
            )


            recommendations = (
                generate_learning_recommendations(
                    sorted(
                        missing_skills
                    )
                )
            )


            if recommendations:

                for recommendation in (
                    recommendations
                ):

                    st.write(
                        "📌",
                        recommendation
                    )

            else:

                st.success(
                    "Your skills match the selected "
                    "job well!"
                )


            # =================================================
            # COMPLETE
            # =================================================

            st.divider()

            st.success(
                "🎉 Resume analysis completed successfully!"
            )