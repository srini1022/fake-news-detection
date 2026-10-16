import csv
import re
import joblib
import yaml
import nltk

from pathlib import Path
from nltk.corpus import stopwords
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Locate and load the saved model artifacts.
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "model_artifacts"

MODEL_PATH = ARTIFACT_DIR / "fake_news_model.pkl"
VECTORIZER_PATH = ARTIFACT_DIR / "tfidf_vectorizer.pkl"
YAML_PATH = BASE_DIR / "test_cases.yaml"
CSV_PATH = BASE_DIR / "external_test_results.csv"

# Stop early with a clear error if a required file is missing.
for path in [MODEL_PATH, VECTORIZER_PATH, YAML_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

# Load the model and vectorizer trained in earlier phases.
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# --------------------------------------------------
# 2. Define the same text preprocessing as app.py.
# --------------------------------------------------

# Download stopwords only if they are not already installed.
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))

# Preserve negation because it can change a sentence's meaning.
stop_words.difference_update({"no", "nor", "not", "never"})


def clean_text(text):
    """Clean article text consistently with the Streamlit app."""

    text = str(text).lower()

    # Remove URLs and HTML tags.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Keep alphabetic words and remove punctuation and digits.
    words = re.findall(r"\b[a-z]+\b", text)

    # Remove English stopwords while preserving negations.
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# --------------------------------------------------
# 3. Read and validate the YAML test cases.
# --------------------------------------------------

with open(YAML_PATH, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file) or {}

cases = data.get("test_cases", [])

if not isinstance(cases, list) or not cases:
    raise ValueError(
        "No test cases found. Add verified articles to test_cases.yaml."
    )

required_fields = {
    "id", "source", "headline", "text", "expected_label"
}

seen_ids = set()

for case in cases:
    if not isinstance(case, dict):
        raise ValueError("Each test case must be a YAML mapping.")

    missing = required_fields - case.keys()

    if missing:
        raise ValueError(
            f"Test case is missing fields: {sorted(missing)}"
        )

    if not str(case["headline"]).strip() or not str(case["text"]).strip():
        raise ValueError(f"Empty headline or text in {case['id']}.")

    if case["expected_label"] not in (0, 1):
        raise ValueError(
            f"Invalid expected_label in {case['id']}. Use 0 or 1."
        )

    if case["id"] in seen_ids:
        raise ValueError(f"Duplicate test case ID: {case['id']}")

    seen_ids.add(case["id"])

# A small test set is useful for an initial check, but results
# will be more informative with 10–20 verified articles.
if len(cases) < 10:
    print(
        f"WARNING: Only {len(cases)} cases found. "
        "Aim for 10–20 independent articles."
    )

if {case["expected_label"] for case in cases} != {0, 1}:
    raise ValueError(
        "Include verified examples from both Fake (0) and Real (1)."
    )


# --------------------------------------------------
# 4. Prepare inputs and run predictions.
# --------------------------------------------------

# Combine headline and article text in the same order as training.
cleaned_texts = [
    clean_text(str(case["headline"]) + " " + str(case["text"]))
    for case in cases
]

if any(not text.strip() for text in cleaned_texts):
    raise ValueError(
        "At least one test case has no usable words after cleaning."
    )

# Transform with the fitted vectorizer; never fit it on test data.
X_external = vectorizer.transform(cleaned_texts)

y_true = [case["expected_label"] for case in cases]

# Predict the class for every independent article.
y_pred = model.predict(X_external)

# Get the estimated probability of the predicted class, if available.
confidences = [None] * len(cases)

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X_external)
    model_classes = list(model.classes_)

    for i, predicted_label in enumerate(y_pred):
        class_index = model_classes.index(predicted_label)
        confidences[i] = float(probabilities[i, class_index])


# --------------------------------------------------
# 5. Save per-article results to a CSV file.
# --------------------------------------------------

with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "id",
        "source",
        "headline",
        "expected_label",
        "predicted_label",
        "confidence",
        "correct",
    ])

    for case, actual, predicted, confidence in zip(
        cases, y_true, y_pred, confidences
    ):
        writer.writerow([
            case["id"],
            case["source"],
            case["headline"],
            actual,
            int(predicted),
            "" if confidence is None else round(confidence, 4),
            actual == predicted,
        ])


# --------------------------------------------------
# 6. Calculate and display evaluation metrics.
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred, pos_label=1, zero_division=0
)

recall = recall_score(
    y_true, y_pred, pos_label=1, zero_division=0
)

f1 = f1_score(
    y_true, y_pred, pos_label=1, zero_division=0
)

print("\n========== EXTERNAL TEST RESULTS ==========")
print(f"Articles tested : {len(cases)}")
print(f"Accuracy        : {accuracy:.4f}")
print(f"Precision (Real): {precision:.4f}")
print(f"Recall (Real)   : {recall:.4f}")
print(f"F1-score (Real) : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        zero_division=0,
    )
)

print("Confusion Matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

# Show individual mistakes so they can be reviewed.
print("\nIncorrect Predictions:")

mistakes = 0

for case, actual, predicted in zip(cases, y_true, y_pred):
    if actual != predicted:
        mistakes += 1
        print(
            f"{case['id']}: expected "
            f"{'Real' if actual == 1 else 'Fake'}, predicted "
            f"{'Real' if predicted == 1 else 'Fake'}"
        )

if mistakes == 0:
    print("No incorrect predictions in this test set.")

print(f"\nCSV report saved to: {CSV_PATH}")