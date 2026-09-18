import streamlit as st
import pandas as pd
import re
from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import sys


project_folder = Path(__file__).resolve().parent.parent
sys.path.append(str(project_folder))

from src.data_processing import load_data
from src.text_preprocessing import clean_text
from src.feature_extraction import create_tfidf_features
from src.job_matcher import find_best_job
from src.skill_gap import convert_skills_to_list, analyze_skill_gap
from src.classifier import train_career_model, predict_career
from src.recommender import generate_learning_recommendations
from src.evaluation import evaluate_model
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



# =========================================================
# CLEAN TEXT
# =========================================================




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
    try:
        reader = PdfReader(uploaded_file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

        if not text.strip():
            raise ValueError("No readable text found in the PDF.")

        return text.strip()

    except Exception as e:
        st.error(
            "Unable to read this PDF. "
            "Please upload a valid PDF containing readable text."
        )
        st.stop()


# =========================================================
# READ TXT
# =========================================================

def extract_txt_text(uploaded_file):
    return uploaded_file.read().decode("utf-8", errors="ignore")


# =========================================================
# EXTRACT SKILLS
# =========================================================

def extract_skills(resume_text, job_df):
    resume_text = resume_text.lower()
    all_skills = []

    for skill_text in job_df["Skills"].dropna():
        skill_text = str(skill_text)
        skill_text = skill_text.replace("|", ",")
        skill_text = skill_text.replace(";", ",")

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
# RESUME UPLOAD
# =========================================================

st.header("📄 Resume Input")

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "txt"],
    key="resume_uploader",
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

            if not resume_text or not resume_text.strip():
                st.error(
                    "No readable text was found in the uploaded resume. "
                    "Please upload a resume containing readable text."
                )
                st.stop()


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

            vectorizer, resume_vectors, job_vectors = create_tfidf_features(
             resume_df["clean_resume_text"],
             job_df["clean_responsibilities"]
            )

            uploaded_vector = vectorizer.transform([clean_resume])

               


            # =================================================
            # 4. JOB MATCHING
            # =================================================

            try:
                job_match_result = find_best_job(
                    uploaded_vector,
                    job_vectors,
                    job_df
                )

            except Exception:
                st.error(
                    "Unable to find a suitable job match. "
                    "Please check the job dataset and try again."
                )
                st.stop()

            # =================================================
            # 5. SKILL GAP ANALYSIS
            # =================================================

            try:
                job_skills = convert_skills_to_list(
                    job_match_result["Skills"]
                )

                skill_gap_result = analyze_skill_gap(
                    detected_skills,
                    job_skills
                )

                matched_skills = set(
                    skill_gap_result["matched_skills"]
                )

                missing_skills = set(
                    skill_gap_result["missing_skills"]
                )

            except Exception:
                st.error(
                    "Unable to calculate the skill gap. "
                    "Please check the job skill information and try again."
                )
                st.stop()


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

            career_model = train_career_model(
                X,
                y
            )

            career_predictions = predict_career(
                career_model,
                uploaded_vector,
                top_n=3
            )


            for prediction in career_predictions:
                st.write(
                    f'**{prediction["Career"]}** — '
                    f'{prediction["Confidence"]:.2f}%'
                )

            # =========================================================
            # 6.1 MODEL EVALUATION
            # =========================================================

            st.subheader("📈 Model Evaluation")

            y_pred = career_model.predict(X)

            evaluation_results = evaluate_model(
                y,
                y_pred
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Accuracy",
                    f'{evaluation_results["accuracy"]:.2f}%'
                )

            with col2:
                st.metric(
                    "Precision",
                    f'{evaluation_results["precision"]:.2f}%'
                )

            with col3:
                st.metric(
                    "Recall",
                    f'{evaluation_results["recall"]:.2f}%'
                )

            with col4:
                st.metric(
                    "F1 Score",
                    f'{evaluation_results["f1_score"]:.2f}%'
                )

            st.write("### Confusion Matrix")

            st.write(
                evaluation_results["confusion_matrix"]
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