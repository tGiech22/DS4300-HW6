#!/usr/bin/env python3
"""Extract skill terms from cleaned postings and update MongoDB."""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

from pymongo import MongoClient, UpdateOne


SKILL_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "programming": {
        "Python": [r"\bpython\b"],
        "SQL": [r"\bsql\b"],
        "PostgreSQL": [r"\bpostgres(?:ql)?\b"],
        "MySQL": [r"\bmysql\b"],
        "R": [r"(?<![A-Za-z0-9])R(?![A-Za-z0-9])"],
        "Java": [r"\bjava\b"],
        "Scala": [r"\bscala\b"],
        "C++": [r"\bc\+\+\b"],
        "Rust": [r"\brust\b"],
    },
    "ml_frameworks": {
        "PyTorch": [r"\bpytorch\b"],
        "TensorFlow": [r"\btensorflow\b"],
        "Keras": [r"\bkeras\b"],
        "Scikit-learn": [r"\bscikit[- ]learn\b", r"\bsklearn\b"],
        "XGBoost": [r"\bxgboost\b"],
    },
    "data_engineering": {
        "Spark": [r"\bspark\b", r"\bapache spark\b"],
        "Hadoop": [r"\bhadoop\b"],
        "Airflow": [r"\bairflow\b"],
        "Kafka": [r"\bkafka\b"],
        "Databricks": [r"\bdatabricks\b"],
        "Ray": [r"\bray\b"],
        "ETL/ELT": [r"\betl\b", r"\belt\b"],
        "Pandas": [r"\bpandas\b"],
    },
    "cloud_platforms": {
        "AWS": [r"\baws\b", r"\bamazon web services\b"],
        "Azure": [r"\bazure\b"],
        "GCP": [r"\bgcp\b", r"\bgoogle cloud\b"],
        "Kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
        "Docker": [r"\bdocker\b"],
        "Terraform": [r"\bterraform\b"],
    },
    "databases": {
        "PostgreSQL": [r"\bpostgres(?:ql)?\b"],
        "MySQL": [r"\bmysql\b"],
        "MongoDB": [r"\bmongodb\b"],
        "Elasticsearch": [r"\belasticsearch\b"],
        "DynamoDB": [r"\bdynamodb\b"],
        "Redis": [r"\bredis\b"],
    },
    "analytics_bi": {
        "Tableau": [r"\btableau\b"],
        "Power BI": [r"\bpower ?bi\b"],
        "Excel": [r"\bexcel\b"],
        "Superset": [r"\bsuperset\b", r"\bapache superset\b"],
        "Plotly": [r"\bplotly\b"],
    },
    "llm_genai": {
        "NLP": [r"\bnlp\b", r"\bnatural language processing\b"],
        "LLM": [r"\bllms?\b", r"\blarge language models?\b"],
        "Transformers": [r"\btransformers?\b"],
        "LangChain": [r"\blangchain\b"],
        "Prompt Engineering": [r"\bprompt engineering\b"],
        "Generative AI": [r"\bgenerative ai\b", r"\bgenai\b"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract skills from cleaned postings and store them in MongoDB."
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        help="MongoDB connection string.",
    )
    parser.add_argument(
        "--db-name",
        default=os.getenv("MONGO_DB", "HW6"),
        help="MongoDB database name.",
    )
    parser.add_argument(
        "--collection",
        default="clean_postings",
        help="Collection containing normalized postings.",
    )
    return parser.parse_args()


def detect_skills(text: str) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {category: [] for category in SKILL_PATTERNS}
    all_skills: List[str] = []

    for category, skill_map in SKILL_PATTERNS.items():
        for canonical_name, patterns in skill_map.items():
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                found[category].append(canonical_name)
                all_skills.append(canonical_name)

    found["all"] = sorted(set(all_skills))
    for category in SKILL_PATTERNS:
        found[category] = sorted(found[category])
    return found


def build_description(doc: Dict) -> str:
    raw = doc.get("raw", {})
    return " ".join(
        part for part in [raw.get("job_title"), raw.get("company_description"), raw.get("job_description_text")] if part
    )


def main() -> None:
    args = parse_args()
    client = MongoClient(args.mongo_uri)
    collection = client[args.db_name][args.collection]

    operations: List[UpdateOne] = []
    updated = 0

    for doc in collection.find({}, {"_id": 1, "raw": 1}):
        text = build_description(doc)
        skills = detect_skills(text)
        operations.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {"skills": skills}},
            )
        )
        updated += 1

    if not operations:
        print("No cleaned postings found. Run clean_postings.py first.")
        return

    collection.bulk_write(operations, ordered=False)
    collection.create_index("skills.all")
    print(f"Extracted skills for {updated} postings.")


if __name__ == "__main__":
    main()
