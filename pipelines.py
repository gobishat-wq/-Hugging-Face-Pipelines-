print("WEEK 7 PROJECT STARTED")
import pandas as pd
from pathlib import Path
from transformers import pipeline

print("STEP 1: IMPORTS COMPLETED")

# --------------------------------
# 1. PROJECT FOLDER LOCATION
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

# Load input dataset
df = pd.read_csv(DATASET_DIR / "input_text.csv")

print("STEP 2: CSV LOADED SUCCESSFULLY")
print("Number of rows:", len(df))

print("\n========== INPUT DATASET ==========")
print(df)

# Store all pipeline comparison results
comparison_results = []


# --------------------------------
# 2. SENTIMENT ANALYSIS
# --------------------------------

print("\nLoading Sentiment Analysis Model...")

sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

sentiment_results = []

for text in df["text"]:
    output = sentiment(text)[0]

    sentiment_results.append({
        "Text": text,
        "Sentiment": output["label"],
        "Score": output["score"]
    })

    comparison_results.append({
        "Task": "Sentiment Analysis",
        "Input": text,
        "Output": output["label"]
    })

sentiment_df = pd.DataFrame(sentiment_results)

sentiment_df.to_csv(
    DATASET_DIR / "sentiment_results.csv",
    index=False
)

print("\n========== SENTIMENT RESULTS ==========")
print(sentiment_df)


# --------------------------------
# 3. TEXT SUMMARIZATION
# --------------------------------

print("\nLoading Summarization Model...")

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1
)

summary_results = []

for text in df["text"]:
    output = summarizer(
        text,
        max_length=60,
        min_length=10,
        do_sample=False,
        truncation=True
    )[0]

    summary = output["summary_text"]

    summary_results.append({
        "Original_Text": text,
        "Summary": summary
    })

    comparison_results.append({
        "Task": "Summarization",
        "Input": text,
        "Output": summary
    })

summary_df = pd.DataFrame(summary_results)

summary_df.to_csv(
    DATASET_DIR / "summarized_text.csv",
    index=False
)

print("\n========== SUMMARY RESULTS ==========")
print(summary_df)


# --------------------------------
# 4. ENGLISH TO TAMIL TRANSLATION
# --------------------------------

print("\nLoading Translation Model...")

translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-ta",
    device=-1
)

translation_results = []

for text in df["text"]:
    output = translator(text, max_length=128)[0]

    translated_text = output["translation_text"]

    translation_results.append({
        "Original_Text": text,
        "Tamil_Translation": translated_text
    })

    comparison_results.append({
        "Task": "Translation",
        "Input": text,
        "Output": translated_text
    })

translation_df = pd.DataFrame(translation_results)

translation_df.to_csv(
    DATASET_DIR / "translated_text.csv",
    index=False
)

print("\n========== TRANSLATION RESULTS ==========")
print(translation_df)


# --------------------------------
# 5. QUESTION ANSWERING
# --------------------------------

print("\nLoading Question Answering Model...")

qa_pipeline = pipeline(
    "question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad",
    device=-1
)

context = """
Artificial Intelligence is a technology that helps computers
perform tasks that normally require human intelligence.
It is used in healthcare, education, banking and business.
Machine learning is a branch of Artificial Intelligence.
It helps computers learn patterns from data and make predictions.
"""

questions = [
    "What is Artificial Intelligence?",
    "Where is AI used?",
    "What is machine learning?",
    "What does machine learning help computers do?",
    "What kind of tasks can AI perform?"
]

qa_results = []

for question in questions:
    output = qa_pipeline(
        question=question,
        context=context
    )

    answer = output["answer"]

    qa_results.append({
        "Context": context,
        "Question": question,
        "Answer": answer,
        "Score": output["score"]
    })

    comparison_results.append({
        "Task": "Question Answering",
        "Input": question,
        "Output": answer
    })

qa_df = pd.DataFrame(qa_results)

qa_df.to_csv(
    DATASET_DIR / "qa_results.csv",
    index=False
)

print("\n========== QUESTION ANSWERING RESULTS ==========")
print(qa_df)


# --------------------------------
# 6. PIPELINE COMPARISON
# --------------------------------

comparison_df = pd.DataFrame(comparison_results)

comparison_df.to_csv(
    DATASET_DIR / "pipeline_comparison.csv",
    index=False
)

print("\n========== PIPELINE COMPARISON ==========")
print(comparison_df)


# --------------------------------
# 7. FINAL OUTPUT
# --------------------------------

print("\n====================================")
print("WEEK 7 HUGGING FACE PROJECT COMPLETED")
print("====================================")

print("\nAll CSV files saved successfully!")
print("Check the dataset folder.")
