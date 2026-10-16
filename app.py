# Import the libraries required for the Streamlit application.
import re
from pathlib import Path

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords


# Configure the browser tab and page layout.
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# Download stopwords if they are not already available.
nltk.download("stopwords", quiet=True)

# Preserve negation words because they can change the meaning of a sentence.
stop_words = set(stopwords.words("english"))
stop_words.difference_update({"no", "nor", "not", "never"})


# Load the saved model and vectorizer only once per application session.
@st.cache_resource
def load_artifacts():
    # Resolve artifact paths relative to this Python file.
    artifact_dir = Path(__file__).parent / "model_artifacts"

    # Load the trained classifier and the fitted TF-IDF vectorizer.
    model = joblib.load(artifact_dir / "fake_news_model.pkl")
    vectorizer = joblib.load(artifact_dir / "tfidf_vectorizer.pkl")

    return model, vectorizer


# Apply the same text-cleaning operations used during model training.
def clean_text(text):
    # Convert text to lowercase for consistent processing.
    text = str(text).lower()

    # Remove web addresses and HTML tags.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Retain alphabetic words and remove punctuation and digits.
    words = re.findall(r"\b[a-z]+\b", text)

    # Remove stopwords while retaining negation words.
    words = [word for word in words if word not in stop_words]

    # Return cleaned text in the format expected by TF-IDF.
    return " ".join(words)


# Display the application heading and its purpose.
st.title("📰 Fake News Detection")
st.write(
    "Enter a news headline and article text to classify the article "
    "as Fake or Real using a trained Machine Learning model."
)

st.info(
    "This tool provides an ML-based prediction. "
    "It does not independently verify news or establish factual truth."
)


# Collect the article information from the user.
title = st.text_input("News Headline")
article_text = st.text_area(
    "News Article",
    height=250,
    placeholder="Paste the news article here..."
)


# Run prediction only when the user clicks the button.
if st.button("Analyze News", type="primary", use_container_width=True):

    # Require both fields so the model receives meaningful input.
    if not title.strip() or not article_text.strip():
        st.warning("Please enter both the headline and article text.")

    else:
        try:
            # Load the existing model and vectorizer.
            model, vectorizer = load_artifacts()

            # Combine the fields in the same order used during training.
            combined_text = title + " " + article_text

            # Apply the training-time text-cleaning procedure.
            cleaned_text = clean_text(combined_text)

            # Check whether preprocessing left any usable words.
            if not cleaned_text.strip():
                st.warning(
                    "No usable English words remain after preprocessing. "
                    "Please enter more meaningful article text."
                )

            else:
                # Transform the input using the fitted vectorizer.
                # Do not fit or retrain the vectorizer on user input.
                features = vectorizer.transform([cleaned_text])

                # Predict the class: 0 = Fake, 1 = Real.
                prediction = int(model.predict(features)[0])

                # Obtain the model's estimated probabilities, if supported.
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    classes = list(model.classes_)

                    # Find the probability corresponding to the predicted class.
                    predicted_index = classes.index(prediction)
                    confidence = float(probabilities[predicted_index])

                else:
                    confidence = None

                # Display the prediction and its estimated confidence.
                st.subheader("Prediction")

                if prediction == 0:
                    st.error("Classification: Fake News")
                else:
                    st.success("Classification: Real News")

                if confidence is not None:
                    st.metric(
                        "Model confidence",
                        f"{confidence:.2%}"
                    )

                    st.caption(
                        "Confidence is the model's estimated probability "
                        "for its predicted class, not a guarantee of accuracy."
                    )

        except FileNotFoundError:
            st.error(
                "Model artifacts were not found. Run the notebook's "
                "artifact-saving cell and check the model_artifacts folder."
            )

        except Exception as error:
            st.error(f"Unable to analyze the article: {error}")