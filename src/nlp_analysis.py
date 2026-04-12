# Assignment: DS4300 Homework 6
# Written by: Erika Sohn
# Function: Performs NLP analysis on job description text from MongoDB clean_postings collection

import os, re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from pymongo import MongoClient
import numpy as np
import argparse

FIGURES_DIR = Path("outputs/figures")
MIN_POSTINGS = 15
TOP_N = 10
EXTRA_STOPWORDS = {
    "experience", "work", "working", "team", "role", "job",
    "skills", "ability", "strong", "knowledge", "using",
    "including", "well", "also", "across", "within", "etc",
    "will", "must", "may", "one", "two", "years", "year",
    "company", "position", "opportunity", "looking", "great",
    "help", "new", "use", "used", "us", "get", "make",
    "meta", "pay", "tiktok", "google", "amazon", "apple", "microsoft", 
}

def fetch_documents(collection) -> list[tuple[str, str]]:
    """
    Fetches job descriptions and role groups from MongoDB
    Returns: list of (role_group, cleaned_text) tuples
    Example: [("Machine Learning Engineer", "python pytorch aws..."), ...]
    """
    # apply ROLE_FILTER to return docs 
    query = {
        "normalized.exclude_for_assignment": False,
        "normalized.exclude_for_role_analysis": False,
        "skills.all": {"$exists": True, "$ne": []},
        "normalized.role_group": {"$ne": None},
        "raw.job_description_text": {"$ne": None},
    }

    # fetch only two fields required
    projection = {
        "normalized.role_group": 1,
        "raw.job_description_text": 1,
        "_id": 0,
    }

    results = []

    # .find() queries the (query) and returns the (projection)
    for doc in collection.find(query, projection):
        role = doc["normalized"]["role_group"]
        text = doc["raw"]["job_description_text"]
        results.append((role, text)) # append tuple 
    return results 

def preprocess_text(text) -> str:
    """
    Lowercases, removes punctuation/numbers, removes stopwords
    Example: "experience pytorch aws kubernetes deployment inference"
    """
    stop_words = set(stopwords.words("english")) | EXTRA_STOPWORDS 

    text = text.lower()                           
    text = re.sub(r"[^a-z\s]", "", text)   

    tokens = []
    for w in text.split():
        if w not in stop_words:
            tokens.append(w)

    # rejoin tokens into single str for NLP input 
    return " ".join(tokens) 

def run_tfidf(docs_by_role) -> dict[str, list[tuple[str, float]]]:
    """
    Runs TF-IDF per role group to find distinctive terms
    Example: {"Machine Learning Engineer": [("inference", 0.42), ("deployment", 0.38)...]}

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html 
    """
    roles = list(docs_by_role.keys())   

    corpus = []
    for role in roles:
        corpus.append(" ".join(docs_by_role[role]))

    # fit TF-IDF acrosss all roles to learn vocab and scores 
    vectorizer = TfidfVectorizer(max_features=5000, min_df=5) # vectorize on top 5000 frequent terms 
    X = vectorizer.fit_transform(corpus)

    # get vocab list and index -> word 
    feature_names = vectorizer.get_feature_names_out()

    # extract top N distinctive terms per role
    results = {}
    for i, role in enumerate(roles):
        # convert sparse matrix row to dense array
        row = X[i].toarray()[0]

        # sort indices highest to lowest score, take top N
        top_indices = row.argsort()[::-1][:TOP_N]

        # map indices back to words with their scores
        top_terms = [(feature_names[j], round(row[j], 4)) for j in top_indices]
        results[role] = top_terms

    return results

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="NLP analysis on job postings.")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default=os.getenv("MONGO_DB", "HW6"))
    parser.add_argument("--collection", default="clean_postings")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    # connect to MongoDB
    client = MongoClient(args.mongo_uri)
    collection = client[args.db_name][args.collection]

    # fetch raw documents from MongoDB
    docs = fetch_documents(collection)
    print(f"Fetched {len(docs)} documents")

    # group preprocessed texts by role, filter out roles below MIN_POSTINGS
    docs_by_role = defaultdict(list)
    for role, text in docs:
        cleaned = preprocess_text(text)
        docs_by_role[role].append(cleaned)

    # filter roles with too few postings for reliable TF-IDF
    docs_by_role = {
        role: texts for role, texts in docs_by_role.items()
        if len(texts) >= MIN_POSTINGS
    }
    print(f"Roles included in TF-IDF: {list(docs_by_role.keys())}")

    # run TFIDF
    tfidf_results = run_tfidf(docs_by_role)

    # save TFIDF results to CSV for visualize.py
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for role, terms in tfidf_results.items():
        for term, score in terms:
            rows.append({"role": role, "term": term, "score": score})

    import csv
    with open(output_dir / "nlp_tfidf.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["role", "term", "score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved nlp_tfidf.csv with {len(rows)} rows to {output_dir}")

if __name__ == "__main__":
    main()
