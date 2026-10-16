
# ============================================================
# PHASE 9: ERROR ANALYSIS AND MODEL IMPROVEMENT
# Fake News Detection Project
# ============================================================

import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import joblib

# ------------------------------------------------------------
# 1. Define project paths
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
ARTIFACT_DIR = os.path.join(BASE_DIR, "model_artifacts")

FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
REAL_PATH = os.path.join(DATA_DIR, "True.csv")
EXTERNAL_PATH = os.path.join(BASE_DIR, "external_test_results.csv")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ------------------------------------------------------------
# 2. Load and prepare the dataset
# ------------------------------------------------------------

fake_df = pd.read_csv(FAKE_PATH)
real_df = pd.read_csv(REAL_PATH)

fake_df["label"] = 0  # Fake news
real_df["label"] = 1  # Real news

data = pd.concat([fake_df, real_df], ignore_index=True)

# Check that required columns exist.
required_columns = {"title", "text", "label"}

if not required_columns.issubset(data.columns):
    raise ValueError(
        "Dataset must contain title, text and label columns."
    )

# Remove missing and empty articles.
data = data.dropna(subset=["title", "text"]).copy()

data["title"] = data["title"].astype(str).str.strip()
data["text"] = data["text"].astype(str).str.strip()

data = data[
    data["title"].ne("") & data["text"].ne("")
].copy()

# Combine title and article body, as in the project.
data["content"] = data["title"] + " " + data["text"]

# Remove exact duplicate articles.
data = data.drop_duplicates(subset=["content"]).reset_index(drop=True)

X = data["content"]
y = data["label"]

print("=" * 55)
print("PHASE 9: ERROR ANALYSIS")
print("=" * 55)
print("Total articles:", len(data))
print("Fake articles:", int((y == 0).sum()))
print("Real articles:", int((y == 1).sum()))

# ------------------------------------------------------------
# 3. Split training and testing data
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining articles:", len(X_train))
print("Testing articles:", len(X_test))

# ------------------------------------------------------------
# 4. Convert text into TF-IDF features
# ------------------------------------------------------------

# Fit the vectorizer only on training data to prevent leakage.
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# ------------------------------------------------------------
# 5. Train baseline and candidate models
# ------------------------------------------------------------

baseline_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=RANDOM_STATE
)

candidate_model = LogisticRegression(
    C=0.5,
    max_iter=1000,
    random_state=RANDOM_STATE
)

print("\nTraining baseline model...")
baseline_model.fit(X_train_vec, y_train)

print("Training candidate model...")
candidate_model.fit(X_train_vec, y_train)

# ------------------------------------------------------------
# 6. Evaluate both models
# ------------------------------------------------------------

def evaluate_model(name, model):
    predictions = model.predict(X_test_vec)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision_Real": precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "Recall_Real": recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "F1_Real": f1_score(
            y_test, predictions, pos_label=1, zero_division=0
        )
    }


results = pd.DataFrame([
    evaluate_model("Baseline (C=1.0)", baseline_model),
    evaluate_model("Candidate (C=0.5)", candidate_model)
])

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
print(results.to_string(index=False))

results.to_csv(
    os.path.join(BASE_DIR, "phase9_model_comparison.csv"),
    index=False
)

# ------------------------------------------------------------
# 7. Select the model using Real-class F1
# ------------------------------------------------------------

baseline_f1 = results.loc[0, "F1_Real"]
candidate_f1 = results.loc[1, "F1_Real"]

if candidate_f1 > baseline_f1:
    selected_model = candidate_model
    selected_name = "Candidate Logistic Regression (C=0.5)"
else:
    selected_model = baseline_model
    selected_name = "Baseline Logistic Regression (C=1.0)"

print("\nSelected model:", selected_name)

# ------------------------------------------------------------
# 8. Analyze held-out test predictions
# ------------------------------------------------------------

test_predictions = selected_model.predict(X_test_vec)

test_results = pd.DataFrame({
    "article": X_test.reset_index(drop=True),
    "actual": y_test.reset_index(drop=True),
    "predicted": test_predictions
})

test_results["correct"] = (
    test_results["actual"] == test_results["predicted"]
)

test_results.to_csv(
    os.path.join(BASE_DIR, "phase9_test_predictions.csv"),
    index=False
)

mistakes = test_results[~test_results["correct"]]

print("\nIncorrect predictions on held-out test data:", len(mistakes))

if not mistakes.empty:
    print("\nSample misclassified articles:")
    print(mistakes[["actual", "predicted"]].head(10).to_string(index=False))

# ------------------------------------------------------------
# 9. Print detailed evaluation
# ------------------------------------------------------------

matrix = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
)

report = classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["Fake", "Real"],
    zero_division=0
)

print("\nConfusion matrix (rows=actual, columns=predicted):")
print(matrix)

print("\nClassification report:")
print(report)

# ------------------------------------------------------------
# 10. Save the Phase 9 report
# ------------------------------------------------------------

report_path = os.path.join(
    BASE_DIR, "phase9_evaluation_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("PHASE 9: ERROR ANALYSIS AND MODEL IMPROVEMENT\n\n")
    file.write("Selected model: " + selected_name + "\n\n")
    file.write(results.to_string(index=False))
    file.write("\n\nConfusion matrix:\n")
    file.write(str(matrix))
    file.write("\n\nClassification report:\n")
    file.write(report)

# ------------------------------------------------------------
# 11. Save separate Phase 9 artifacts
# ------------------------------------------------------------

# Save the selected model and its matching TF-IDF vectorizer.
# Existing app artifacts are not overwritten.

joblib.dump(
    selected_model,
    os.path.join(ARTIFACT_DIR, "phase9_candidate_model.pkl")
)

joblib.dump(
    tfidf,
    os.path.join(ARTIFACT_DIR, "phase9_candidate_tfidf.pkl")
)

print("\n" + "=" * 55)
print("PHASE 9 COMPLETED")
print("=" * 55)
print("Generated files:")
print("- phase9_model_comparison.csv")
print("- phase9_test_predictions.csv")
print("- phase9_evaluation_report.txt")
print("- model_artifacts/phase9_candidate_model.pkl")
print("- model_artifacts/phase9_candidate_tfidf.pkl")
print("\nOriginal application artifacts were not overwritten.")

# ------------------------------------------------------------
# 12. External test error summary
# ------------------------------------------------------------

if os.path.exists(EXTERNAL_PATH):
    external_df = pd.read_csv(EXTERNAL_PATH)

    if {"expected", "predicted"}.issubset(external_df.columns):
        external_mistakes = external_df[
            external_df["expected"] != external_df["predicted"]
        ]

        external_mistakes.to_csv(
            os.path.join(
                BASE_DIR, "phase9_misclassified_articles.csv"
            ),
            index=False
        )

        print(
            "\nExternal test mistakes:",
            len(external_mistakes),
            "out of",
            len(external_df)
        )
        print("Saved: phase9_misclassified_articles.csv")
