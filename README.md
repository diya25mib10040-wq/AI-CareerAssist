# AI-CareerAssist

## AI-Based Resume Analysis, Job Matching and Career Recommendation System

AI-CareerAssist is an AI/ML-based career assistance system that analyzes a user's resume and provides personalized career-related insights.

The system uses Natural Language Processing (NLP), TF-IDF feature extraction, cosine similarity, and Logistic Regression to analyze resume information and generate job, skill, career, and learning recommendations.

---

## 📌 Problem Statement

Students and job seekers often face difficulty in understanding:

- Which jobs are suitable for their current skills
- Which skills are missing from their resume
- Which career categories match their profile
- What skills they should learn next

AI-CareerAssist addresses these problems by analyzing resume content and providing automated career-related recommendations through an interactive Streamlit application.

---

## 🎯 Objectives

The main objectives of AI-CareerAssist are:

1. To extract useful information from a resume.
2. To preprocess and clean resume text.
3. To convert resume text into numerical features using TF-IDF.
4. To identify the most suitable job based on resume-job similarity.
5. To identify matched and missing skills.
6. To predict suitable career categories using Machine Learning.
7. To recommend learning resources for missing skills.
8. To evaluate the career classification model.
9. To provide all results through an easy-to-use web interface.

---

## 🚀 Features

### 1. Resume Upload

Users can upload their resume in:

- PDF format
- TXT format

The system extracts readable text from the uploaded resume.

### 2. Resume Text Preprocessing

The extracted resume text is cleaned before analysis.

The preprocessing stage prepares the text for feature extraction and machine learning.

### 3. TF-IDF Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text into numerical feature vectors.

These vectors are used for comparing resume content with job information.

### 4. Job Matching

The system compares the uploaded resume with available job descriptions using cosine similarity.

The system displays:

- Best matching job
- Match score
- Experience level
- Required skills

### 5. Skill Gap Analysis

The system identifies:

- Skills already present in the resume
- Skills required for the matched job
- Missing skills

This helps users understand which skills they need to develop.

### 6. Career Classification

A Logistic Regression model is used to classify the resume into suitable career categories.

The system displays the top career predictions with their confidence values.

### 7. Learning Recommendations

Learning recommendations are generated based on the missing skills.

Examples include:

- Python
- Machine Learning
- SQL
- Data Analysis
- Deep Learning
- Git
- GitHub
- Cloud Computing
- Docker

### 8. Model Evaluation

The career classification model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| Pandas | Data processing |
| Scikit-learn | Machine Learning and NLP |
| TF-IDF | Text feature extraction |
| Cosine Similarity | Resume-job matching |
| Logistic Regression | Career classification |
| PyPDF | PDF text extraction |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualization where applicable |

---

## 📂 Project Structure

```text
AI-CareerAssist/
│
├── app/
│   ├── app.py
│   └── app copy.py
│
├── data/
│   ├── Resume.csv
│   ├── job_dataset.csv
│   └── final_career_analysis.csv
│
├── notebooks/
│   └── 01_data_analysis.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── text_preprocessing.py
│   ├── feature_extraction.py
│   ├── job_matcher.py
│   ├── skill_gap.py
│   ├── classifier.py
│   ├── recommender.py
│   └── evaluation.py
│
├── tests/
│   ├── test_skill_gap.py
│   ├── test_job_matcher.py
│   ├── test_classifier.py
│   ├── test_evaluation.py
│   └── test_recommender.py
│
├── docs/
│   ├── architecture_diagram.png
│   ├── process_flow_diagram.png
│   └── uml_diagram.png
│
├── ML_EVALUATION.md
└── README.md
🔄 System Workflow
Resume Upload
      ↓
Resume Text Extraction
      ↓
Text Preprocessing
      ↓
TF-IDF Feature Extraction
      ↓
Job Matching
      ↓
Skill Detection
      ↓
Skill Gap Analysis
      ↓
Career Classification
      ↓
Learning Recommendations
      ↓
Model Evaluation
      ↓
Display Results
📊 Machine Learning Approach
Job Matching

Resume and job information are converted into TF-IDF vectors.

Cosine similarity is then used to measure the similarity between the resume and available jobs.

A higher similarity score indicates greater textual similarity between the resume and the job information.

Career Classification

The career classification module uses:

TF-IDF Features
       ↓
Logistic Regression
       ↓
Career Category Prediction

The system generates the top career categories using the predicted class probabilities.

📈 Model Evaluation

The project calculates:

Accuracy

Measures the proportion of correctly classified samples.

Precision

Measures how many predicted samples of a class are actually members of that class.

Recall

Measures how many samples belonging to a class are correctly identified.

F1 Score

Combines precision and recall into a single metric.

Confusion Matrix

Shows the relationship between actual and predicted career categories.

Note: The current application evaluates the trained career classification model using the same feature matrix used during training. Therefore, the displayed evaluation metrics represent training-data performance and should not be interpreted as final unseen-data performance.

A future version can use a train-test split or cross-validation for more reliable evaluation.

🧪 Testing

The project contains unit tests for important modules.

Tested modules include:

Skill gap analysis
Job matching
Career classification
Model evaluation
Learning recommendations

The test suite currently contains 7 tests.

Run all tests using:

python -m pytest tests/

Expected successful result:

7 passed
🖥️ Running the Application
Step 1: Clone or download the project

Place the project on your local computer.

Step 2: Open the project in VS Code

Open the AI-CareerAssist folder.

Step 3: Activate the virtual environment

On Windows PowerShell:

.\.venv\Scripts\Activate.ps1
Step 4: Install required libraries
pip install streamlit pandas scikit-learn pypdf numpy matplotlib seaborn pytest
Step 5: Run the Streamlit application
streamlit run app/app.py

The application will open in your web browser.

📥 Input

The primary user input is:

Resume in PDF or TXT format

The system also uses job and resume datasets stored in the data folder.

📤 Output

After analysis, the application provides:

Resume analysis
Detected skills
Best job match
Job match score
Required skills
Matched skills
Missing skills
Career recommendations
Learning recommendations
Model evaluation metrics
Confusion matrix
🔮 Future Scope

The system can be improved by:

Using larger and more diverse datasets.
Using train-test split and cross-validation.
Adding more career categories.
Improving skill extraction using advanced NLP techniques.
Adding semantic embeddings for better resume-job matching.
Adding personalized learning platforms and course links.
Supporting more resume formats.
Adding explainable AI for career predictions.
Improving recommendation accuracy using advanced ML models.
Deploying the application as a cloud-based service.
⚠️ Limitations

The current system has some limitations:

Job matching primarily depends on textual similarity.
Skill detection depends on skills available in the job dataset.
Career classification depends on the available training data.
Learning recommendations are based on a predefined learning-resource mapping.
Current model evaluation uses training data rather than an independent test set.
👩‍💻 Project

Project Name: AI-CareerAssist

Domain: Artificial Intelligence and Machine Learning

Application Type: Streamlit Web Application

Main Areas: NLP, Machine Learning, Resume Analysis, Job Matching and Career Recommendation


### Where to put it

In VS Code:

**Right-click `AI-CareerAssist` → New File → `README.md`**

Then paste the complete content above and press **Ctrl + S**.

Your project root should look like:

```text
AI-CareerAssist
├── README.md             ← ⭐ NEW
├── ML_EVALUATION.md
├── app
├── data
├── docs
├── notebooks
├── src
└── tests