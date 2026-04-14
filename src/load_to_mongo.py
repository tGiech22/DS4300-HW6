#!/usr/bin/env python3
"""Load supported job postings CSV files into MongoDB."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from pymongo import MongoClient, UpdateOne


DEFAULT_CSV = Path(__file__).resolve().parents[1] / "Job_Postings_US new.csv"
ORIGINAL_SCHEMA = {
    "ID",
    "job_posted_date",
    "company_address_locality",
    "company_address_region",
    "company_name",
    "company_description",
    "job_description_text",
    "seniority_level",
    "job_title",
}
LINKEDIN_SCHEMA = {
    "title",
    "location",
    "publishedAt",
    "companyName",
    "description",
    "experienceLevel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the raw job postings CSV into a MongoDB collection."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to the source CSV file.",
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
        default="raw_postings",
        help="Target collection for the raw postings.",
    )
    parser.add_argument(
        "--source-name",
        default="Job_Postings_US new.csv",
        help="Source label stored with each inserted document.",
    )
    return parser.parse_args()


def clean_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if normalized else None


def detect_schema(fieldnames: Sequence[str]) -> str:
    columns = set(fieldnames)
    if ORIGINAL_SCHEMA.issubset(columns):
        return "original"
    if LINKEDIN_SCHEMA.issubset(columns):
        return "linkedin"
    raise ValueError(
        "Unsupported CSV schema. Expected either the original postings CSV "
        "or ai_ml_jobs_linkedin.csv."
    )


def split_location(location: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    value = clean_value(location)
    if value is None:
        return None, None
    if "," not in value:
        return value, None
    city, region = value.rsplit(",", 1)
    return clean_value(city), clean_value(region)


def normalize_original_row(
    row: Dict[str, Optional[str]], source_name: str
) -> Dict[str, Optional[str]]:
    source_id = clean_value(row.get("ID"))
    return {
        "source_name": source_name,
        "source_id": source_id,
        "job_posted_date": clean_value(row.get("job_posted_date")),
        "company_address_locality": clean_value(row.get("company_address_locality")),
        "company_address_region": clean_value(row.get("company_address_region")),
        "company_name": clean_value(row.get("company_name")),
        "company_description": clean_value(row.get("company_description")),
        "job_description_text": clean_value(row.get("job_description_text")),
        "seniority_level": clean_value(row.get("seniority_level")),
        "job_title": clean_value(row.get("job_title")),
    }


def normalize_linkedin_row(
    row: Dict[str, Optional[str]], source_name: str, row_number: int
) -> Dict[str, Optional[str]]:
    city, region = split_location(row.get("location"))
    description = clean_value(row.get("description"))
    sector = clean_value(row.get("sector"))
    company_description = sector if sector else None
    source_id = f"linkedin_{row_number}"
    return {
        "source_name": source_name,
        "source_id": source_id,
        "job_posted_date": clean_value(row.get("publishedAt")),
        "company_address_locality": city,
        "company_address_region": region,
        "company_name": clean_value(row.get("companyName")),
        "company_description": company_description,
        "job_description_text": description,
        "seniority_level": clean_value(row.get("experienceLevel")),
        "job_title": clean_value(row.get("title")),
    }


def read_rows(csv_path: Path, source_name: str) -> Iterable[Dict[str, Optional[str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        schema = detect_schema(reader.fieldnames or [])
        for row_number, row in enumerate(reader, start=1):
            if schema == "original":
                yield normalize_original_row(row, source_name)
            elif schema == "linkedin":
                yield normalize_linkedin_row(row, source_name, row_number)


def build_operations(
    rows: Iterable[Dict[str, Optional[str]]],
) -> List[UpdateOne]:
    operations: List[UpdateOne] = []
    for row in rows:
        source_id = row.get("source_id")
        if source_id is None:
            continue
        operations.append(
            UpdateOne(
                {"source_name": row["source_name"], "source_id": source_id},
                {"$set": row},
                upsert=True,
            )
        )
    return operations


def ensure_indexes(collection) -> None:
    collection.create_index([("source_name", 1), ("source_id", 1)], unique=True)
    collection.create_index("job_posted_date")
    collection.create_index("job_title")
    collection.create_index("company_address_region")


def main() -> None:
    args = parse_args()
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    rows = list(read_rows(args.csv_path, args.source_name))
    operations = build_operations(rows)

    client = MongoClient(args.mongo_uri)
    collection = client[args.db_name][args.collection]
    ensure_indexes(collection)

    if operations:
        result = collection.bulk_write(operations, ordered=False)
        print(f"Loaded documents into {args.db_name}.{args.collection}")
        print(f"Matched existing: {result.matched_count}")
        print(f"Modified existing: {result.modified_count}")
        print(f"Upserted new: {len(result.upserted_ids)}")
    else:
        print("No rows were loaded.")


if __name__ == "__main__":
    main()
