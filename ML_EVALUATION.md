# ML Model Evaluation

## 1. Purpose

The Career Classification module uses Logistic Regression to predict suitable career categories from resume text.

The model is evaluated using four classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score

A confusion matrix is also generated to analyze the classification results.

---

## 2. Model Used

The project uses Logistic Regression for career classification.

The model is trained using TF-IDF features extracted from resume text.

### Pipeline

Resume Text
↓
Text Preprocessing
↓
TF-IDF Feature Extraction
↓
Logistic Regression
↓
Career Prediction

---

## 3. Evaluation Metrics

### Accuracy

Accuracy measures the proportion of correctly classified resumes among all evaluated resumes.

Formula:

Accuracy = Correct Predictions / Total Predictions

A higher accuracy means more samples were classified correctly.

---

### Precision

Precision measures how many samples predicted as a particular career category actually belong to that category.

It is useful when false positive predictions need to be considered.

---

### Recall

Recall measures how many samples belonging to a particular career category were correctly identified by the model.

It is useful when missing relevant career categories is important.

---

### F1 Score

F1 Score combines Precision and Recall into a single metric.

Formula:

F1 Score = 2 × (Precision × Recall) / (Precision + Recall)

A higher F1 Score indicates a better balance between precision and recall.

---

## 4. Confusion Matrix

The confusion matrix shows the relationship between actual career categories and predicted career categories.

- Rows represent actual classes.
- Columns represent predicted classes.
- Diagonal values represent correct predictions.
- Off-diagonal values represent misclassifications.

---

## 5. Implementation

The project uses the `evaluate_model()` function from:

`src/evaluation.py`

The function calculates:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- Confusion Matrix

---

## 6. Current Evaluation Method

The current Streamlit application evaluates the trained career classification model using the same feature matrix used during training.

Therefore, the displayed metrics represent performance on the training data.

These values should not be interpreted as final unseen-data or real-world accuracy.

---

## 7. Future Improvement

For a more reliable evaluation, the project can be extended using:

- Train-test split
- Cross-validation
- Separate validation data
- Testing on unseen resumes
- Additional classification metrics
- Hyperparameter tuning