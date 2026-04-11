"""Analyze and generate patterns & statistical findings of
 job posting skills and relationships between other variables for csv output"""

from __future__ import annotations

import argparse
import csv
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List

from pymongo import MongoClient

SELECTED_TREND_SKILLS = ["Python", "SQL", "AWS", "PyTorch", "LLM"]
SELECTED_ROLE_SKILLS = ["Python", "SQL", "AWS", "PyTorch", "TensorFlow", "Docker", "Kubernetes", "LLM"]
SELECTED_SENIORITY_SKILLS = ["Python", "SQL", "AWS", "Docker", "Kubernetes", "PyTorch", "LLM"]
SELECTED_STATE_SKILLS = ["AWS", "Azure", "GCP", "PyTorch", "TensorFlow", "Docker", "Kubernetes"]

BASE_FILTER = {
    "normalized.exclude_for_assignment": False,
    "skills.all": {"$exists": True, "$ne": []},
}

ROLE_FILTER = {
    "normalized.exclude_for_assignment": False,
    "normalized.exclude_for_role_analysis": False,
    "skills.all": {"$exists": True, "$ne": []},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze skill demand from MongoDB.")
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
        help="Collection containing enriched postings.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis",
        help="Directory for exported CSV/JSON outputs.",
    )
    return parser.parse_args()


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def top_skills(collection) -> List[Dict[str, Any]]:
    pipeline = [
        {"$match": BASE_FILTER},
        {"$unwind": "$skills.all"},
        {"$group": {"_id": "$skills.all", "posting_count": {"$sum": 1}}},
        {"$sort": {"posting_count": -1, "_id": 1}},
    ]
    total_docs = collection.count_documents(BASE_FILTER)
    results = []
    for doc in collection.aggregate(pipeline):
        results.append(
            {
                "skill": doc["_id"],
                "posting_count": doc["posting_count"],
                "posting_share": round(doc["posting_count"] / total_docs, 4) if total_docs else 0.0,
            }
        )
    return results


def monthly_skill_trends(collection, selected_skills: List[str]) -> List[Dict[str, Any]]:
    pipeline = [
        {"$match": {**BASE_FILTER, "normalized.posted_month": {"$ne": None}}},
        {
            "$project": {
                "posted_month": "$normalized.posted_month",
                "skills_all": "$skills.all",
            }
        },
        {
            "$group": {
                "_id": "$posted_month",
                "total_postings": {"$sum": 1},
                **{
                    skill: {
                        "$sum": {
                            "$cond": [{"$in": [skill, "$skills_all"]}, 1, 0]
                        }
                    }
                    for skill in selected_skills
                },
            }
        },
        {"$sort": {"_id": 1}},
    ]

    rows = []
    for doc in collection.aggregate(pipeline):
        month = doc["_id"]
        total = doc["total_postings"]
        for skill in selected_skills:
            count = doc.get(skill, 0)
            rows.append(
                {
                    "posted_month": month,
                    "skill": skill,
                    "posting_count": count,
                    "posting_share": round(count / total, 4) if total else 0.0,
                    "monthly_total_postings": total,
                }
            )
    return rows


def role_skill_matrix(collection, selected_skills: List[str]) -> List[Dict[str, Any]]:
    pipeline = [
        {"$match": ROLE_FILTER},
        {
            "$group": {
                "_id": "$normalized.role_group",
                "total_postings": {"$sum": 1},
                **{
                    skill: {
                        "$sum": {
                            "$cond": [{"$in": [skill, "$skills.all"]}, 1, 0]
                        }
                    }
                    for skill in selected_skills
                },
            }
        },
        {"$sort": {"_id": 1}},
    ]

    rows = []
    for doc in collection.aggregate(pipeline):
        role_group = doc["_id"]
        total = doc["total_postings"]
        for skill in selected_skills:
            count = doc.get(skill, 0)
            rows.append(
                {
                    "role_group": role_group,
                    "skill": skill,
                    "posting_count": count,
                    "posting_share": round(count / total, 4) if total else 0.0,
                    "role_total_postings": total,
                }
            )
    return rows


def seniority_skill_matrix(collection, selected_skills: List[str]) -> List[Dict[str, Any]]:
    pipeline = [
        {"$match": BASE_FILTER},
        {
            "$group": {
                "_id": "$normalized.seniority_group",
                "total_postings": {"$sum": 1},
                **{
                    skill: {
                        "$sum": {
                            "$cond": [{"$in": [skill, "$skills.all"]}, 1, 0]
                        }
                    }
                    for skill in selected_skills
                },
            }
        },
        {"$sort": {"_id": 1}},
    ]

    rows = []
    for doc in collection.aggregate(pipeline):
        seniority_group = doc["_id"]
        total = doc["total_postings"]
        for skill in selected_skills:
            count = doc.get(skill, 0)
            rows.append(
                {
                    "seniority_group": seniority_group,
                    "skill": skill,
                    "posting_count": count,
                    "posting_share": round(count / total, 4) if total else 0.0,
                    "seniority_total_postings": total,
                }
            )
    return rows


def state_skill_matrix(collection, selected_skills: List[str], min_postings: int = 10) -> List[Dict[str, Any]]:
    pipeline = [
        {"$match": {**BASE_FILTER, "normalized.state": {"$ne": None}}},
        {
            "$group": {
                "_id": "$normalized.state",
                "total_postings": {"$sum": 1},
                **{
                    skill: {
                        "$sum": {
                            "$cond": [{"$in": [skill, "$skills.all"]}, 1, 0]
                        }
                    }
                    for skill in selected_skills
                },
            }
        },
        {"$match": {"total_postings": {"$gte": min_postings}}},
        {"$sort": {"total_postings": -1, "_id": 1}},
    ]

    rows = []
    for doc in collection.aggregate(pipeline):
        state = doc["_id"]
        total = doc["total_postings"]
        for skill in selected_skills:
            count = doc.get(skill, 0)
            rows.append(
                {
                    "state": state,
                    "skill": skill,
                    "posting_count": count,
                    "posting_share": round(count / total, 4) if total else 0.0,
                    "state_total_postings": total,
                }
            )
    return rows


def skill_pairs(collection, min_pair_count: int = 5) -> List[Dict[str, Any]]:
    pair_counts: Dict[tuple[str, str], int] = {}
    for doc in collection.find(BASE_FILTER, {"skills.all": 1, "_id": 0}):
        skills = sorted(set(doc.get("skills", {}).get("all", [])))
        for a, b in combinations(skills, 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    rows = [
        {"skill_a": a, "skill_b": b, "pair_count": count}
        for (a, b), count in pair_counts.items()
        if count >= min_pair_count
    ]
    rows.sort(key=lambda x: (-x["pair_count"], x["skill_a"], x["skill_b"]))
    return rows


def build_summary(collection) -> Dict[str, Any]:
    return {
        "total_clean_postings": collection.count_documents({}),
        "usable_for_analysis": collection.count_documents(BASE_FILTER),
        "usable_for_role_analysis": collection.count_documents(ROLE_FILTER),
        "excluded_for_assignment": collection.count_documents({"normalized.exclude_for_assignment": True}),
        "excluded_for_role_analysis": collection.count_documents({"normalized.exclude_for_role_analysis": True}),
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    client = MongoClient(args.mongo_uri)
    collection = client[args.db_name][args.collection]

    summary = build_summary(collection)
    top = top_skills(collection)
    trends = monthly_skill_trends(collection, SELECTED_TREND_SKILLS)
    role_matrix = role_skill_matrix(collection, SELECTED_ROLE_SKILLS)
    seniority_matrix = seniority_skill_matrix(collection, SELECTED_SENIORITY_SKILLS)
    state_matrix = state_skill_matrix(collection, SELECTED_STATE_SKILLS, min_postings=10)
    pairs = skill_pairs(collection, min_pair_count=5)

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "top_skills.csv", top)
    write_csv(output_dir / "monthly_skill_trends.csv", trends)
    write_csv(output_dir / "role_skill_matrix.csv", role_matrix)
    write_csv(output_dir / "seniority_skill_matrix.csv", seniority_matrix)
    write_csv(output_dir / "state_skill_matrix.csv", state_matrix)
    write_csv(output_dir / "skill_pairs.csv", pairs)

    print("Analysis exports complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()