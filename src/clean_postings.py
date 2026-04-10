#!/usr/bin/env python3
"""Normalize raw postings and write enriched documents back to MongoDB."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, UpdateOne


STATE_MAP = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "ca": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "ma": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "nj": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "ny": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}


ROLE_PATTERNS = [
    ("Machine Learning Engineer", [r"\bmachine learning engineer\b"]),
    (
        "ML Software Engineer",
        [
            r"\bsoftware engineer,\s*machine learning\b",
            r"\bai/ml software engineer\b",
            r"\bml software engineer\b",
        ],
    ),
    (
        "Data Scientist",
        [r"\bdata scientist\b", r"\bartificial intelligence / data scientist\b"],
    ),
    ("Applied Scientist", [r"\bapplied scientist\b"]),
    (
        "AI Engineer",
        [
            r"\bai engineer\b",
            r"\bartificial intelligence engineer\b",
            r"\bml engineer\b",
        ],
    ),
]


SPORTS_OR_GAMES_PATTERNS = [
    r"\bsports?\b",
    r"\bfitness\b",
    r"\bgaming\b",
    r"\bvideo game\b",
    r"\bcasino\b",
    r"\bbetting\b",
    r"\besports\b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw job postings and store cleaned documents."
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        help="MongoDB connection string.",
    )
    parser.add_argument(
        "--db-name",
        default=os.getenv("MONGO_DB", "ds4300_hw6"),
        help="MongoDB database name.",
    )
    parser.add_argument(
        "--raw-collection",
        default="raw_postings",
        help="Collection containing unmodified postings.",
    )
    parser.add_argument(
        "--clean-collection",
        default="clean_postings",
        help="Collection that will receive cleaned postings.",
    )
    return parser.parse_args()


def normalize_whitespace(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return re.sub(r"\s+", " ", value).strip()


def normalize_state(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in STATE_MAP:
        return STATE_MAP[normalized]
    if len(value.strip()) == 2 and value.strip().isalpha():
        return value.strip().upper()
    return None


def parse_posted_date(value: Optional[str]) -> Dict[str, Optional[str]]:
    if not value:
        return {"posted_date": None, "posted_month": None, "posted_year": None}
    parsed = datetime.strptime(value, "%Y-%m-%d")
    return {
        "posted_date": parsed.date().isoformat(),
        "posted_month": parsed.strftime("%Y-%m"),
        "posted_year": parsed.strftime("%Y"),
    }


def role_group(job_title: Optional[str]) -> str:
    title = (job_title or "").strip().lower()
    for group, patterns in ROLE_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, title):
                return group
    if "engineer" in title and ("ai" in title or "ml" in title):
        return "AI Engineer"
    if "data" in title:
        return "Other AI/Data"
    return "Other AI/Data"


def seniority_group(value: Optional[str]) -> str:
    normalized = (value or "").strip()
    if normalized in {"Internship", "Entry level", "Associate", "Mid-Senior level"}:
        return normalized
    if normalized in {"Director", "Executive"}:
        return "Director/Executive"
    if not normalized or normalized == "Not Applicable":
        return "Unknown/Other"
    return "Unknown/Other"


def infer_exclusion(company_description: Optional[str], job_description: Optional[str]) -> bool:
    combined = " ".join(
        filter(None, [company_description or "", job_description or ""])
    ).lower()
    return any(re.search(pattern, combined) for pattern in SPORTS_OR_GAMES_PATTERNS)


def cleaned_document(raw_doc: Dict[str, Any]) -> Dict[str, Any]:
    company_desc = normalize_whitespace(raw_doc.get("company_description"))
    job_desc = normalize_whitespace(raw_doc.get("job_description_text"))
    date_bits = parse_posted_date(raw_doc.get("job_posted_date"))

    return {
        "source_name": raw_doc.get("source_name"),
        "source_id": raw_doc.get("source_id"),
        "raw": {
            "job_posted_date": raw_doc.get("job_posted_date"),
            "company_name": normalize_whitespace(raw_doc.get("company_name")),
            "company_address_locality": normalize_whitespace(
                raw_doc.get("company_address_locality")
            ),
            "company_address_region": normalize_whitespace(
                raw_doc.get("company_address_region")
            ),
            "company_description": company_desc,
            "job_title": normalize_whitespace(raw_doc.get("job_title")),
            "seniority_level": normalize_whitespace(raw_doc.get("seniority_level")),
            "job_description_text": job_desc,
        },
        "normalized": {
            **date_bits,
            "state": normalize_state(raw_doc.get("company_address_region")),
            "city": normalize_whitespace(raw_doc.get("company_address_locality")),
            "role_group": role_group(raw_doc.get("job_title")),
            "seniority_group": seniority_group(raw_doc.get("seniority_level")),
            "exclude_for_assignment": infer_exclusion(company_desc, job_desc),
        },
        "skills": {
            "programming": [],
            "ml_frameworks": [],
            "data_engineering": [],
            "cloud_platforms": [],
            "databases": [],
            "analytics_bi": [],
            "llm_genai": [],
            "all": [],
        },
    }


def ensure_indexes(collection) -> None:
    collection.create_index([("source_name", 1), ("source_id", 1)], unique=True)
    collection.create_index("normalized.posted_month")
    collection.create_index("normalized.state")
    collection.create_index("normalized.role_group")
    collection.create_index("normalized.exclude_for_assignment")


def main() -> None:
    args = parse_args()
    client = MongoClient(args.mongo_uri)
    raw_collection = client[args.db_name][args.raw_collection]
    clean_collection = client[args.db_name][args.clean_collection]
    ensure_indexes(clean_collection)

    operations: List[UpdateOne] = []
    for raw_doc in raw_collection.find({}, {"_id": 0}):
        clean_doc = cleaned_document(raw_doc)
        operations.append(
            UpdateOne(
                {
                    "source_name": clean_doc["source_name"],
                    "source_id": clean_doc["source_id"],
                },
                {"$set": clean_doc},
                upsert=True,
            )
        )

    if not operations:
        print("No raw postings found. Run load_to_mongo.py first.")
        return

    result = clean_collection.bulk_write(operations, ordered=False)
    print(f"Cleaned documents written to {args.db_name}.{args.clean_collection}")
    print(f"Matched existing: {result.matched_count}")
    print(f"Modified existing: {result.modified_count}")
    print(f"Upserted new: {len(result.upserted_ids)}")


if __name__ == "__main__":
    main()
