"""Generate insightful visuals from skill analysis"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd


DEFAULT_TREND_SKILLS = ["Python", "SQL", "AWS", "PyTorch", "LLM"]

DEFAULT_SENIORITY_SKILLS = ["SQL", "AWS", "Docker", "PyTorch", "LLM"]

SENIORITY_ORDER = [
    "Internship",
    "Entry level",
    "Associate",
    "Mid-Senior level",
    "Director/Executive",
]

DEFAULT_HEATMAP_SKILLS = [
    "Python",
    "SQL",
    "AWS",
    "Docker",
    "Kubernetes",
    "PyTorch",
    "TensorFlow",
    "LLM",
]
DEFAULT_STATE_SKILLS = ["AWS", "Azure", "GCP", "Docker", "Kubernetes"]

ROLE_ORDER = [
    "Machine Learning Engineer",
    "ML Software Engineer",
    "ML Platform / MLOps / Infrastructure",
    "Data Scientist",
    "Data / ML Engineer",
    "Applied Scientist / Research",
    "AI Engineer",
    "Other AI/Data",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visualizations from analysis CSV outputs."
    )
    parser.add_argument(
        "--analysis-dir",
        default="outputs/analysis",
        help="Directory containing analysis CSV outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        default="outputs/figures",
        help="Directory where figures will be saved.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of skills to show in the top-skills chart.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_csv(analysis_dir: Path, filename: str) -> pd.DataFrame:
    path = analysis_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def plot_top_skills(df: pd.DataFrame, out_dir: Path, top_n: int = 15) -> None:
    chart_df = (
        df.sort_values(["posting_share", "posting_count"], ascending=[False, False])
        .head(top_n)
        .sort_values("posting_share", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(chart_df["skill"], chart_df["posting_share"])
    ax.set_title("Top Skills in U.S. AI/ML Job Postings")
    ax.set_xlabel("Share of postings")
    ax.set_ylabel("Skill")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    max_val = chart_df["posting_share"].max()
    ax.set_xlim(0, max_val + 0.18)

    for i, (_, row) in enumerate(chart_df.iterrows()):
        ax.text(
            row["posting_share"] + 0.005,
            i,
            f'{row["posting_share"]:.1%} (n={int(row["posting_count"])})',
            va="center",
            fontsize=9,
        )

    save_figure(fig, out_dir, "01_top_skills")


def plot_skill_timeline(
    df: pd.DataFrame,
    out_dir: Path,
    selected_skills: List[str],
    min_monthly_postings: int = 8,
    smoothing_window: int = 2,
) -> None:
    chart_df = df[df["skill"].isin(selected_skills)].copy()
    chart_df = chart_df[chart_df["monthly_total_postings"] >= min_monthly_postings].copy()

    chart_df["posted_month"] = pd.to_datetime(chart_df["posted_month"], format="%Y-%m")
    chart_df = chart_df.sort_values(["posted_month", "skill"])

    pivot = chart_df.pivot(
        index="posted_month",
        columns="skill",
        values="posting_share",
    )

    ordered_cols = [skill for skill in selected_skills if skill in pivot.columns]
    pivot = pivot[ordered_cols]

    # Light smoothing only
    pivot = pivot.rolling(window=smoothing_window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 6))
    for skill in pivot.columns:
        ax.plot(pivot.index, pivot[skill], marker="o", linewidth=2, label=skill)

    ax.set_title("Skill Demand Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Share of postings ({smoothing_window}-month rolling average)")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    save_figure(fig, out_dir, "02_skill_timeline")


def plot_role_heatmap(
    df: pd.DataFrame, out_dir: Path, selected_skills: List[str]
) -> None:
    chart_df = df[df["skill"].isin(selected_skills)].copy()

    pivot = chart_df.pivot(
        index="role_group",
        columns="skill",
        values="posting_share",
    ).fillna(0)

    role_counts = (
        chart_df.groupby("role_group", as_index=False)["role_total_postings"]
        .max()
        .set_index("role_group")["role_total_postings"]
        .to_dict()
    )

    ordered_roles = [role for role in ROLE_ORDER if role in pivot.index] + [
        role for role in pivot.index if role not in ROLE_ORDER
    ]
    ordered_skills = [skill for skill in selected_skills if skill in pivot.columns]

    pivot = pivot.loc[ordered_roles, ordered_skills]

    row_labels = [f"{role} (n={int(role_counts.get(role, 0))})" for role in pivot.index]

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_title("Skill Demand by Role Group")
    ax.set_xlabel("Skill")
    ax.set_ylabel("Role group")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(row_labels)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            text_color = "white" if val >= 0.45 else "black"
            ax.text(
                j,
                i,
                f"{val:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, label="Share of postings")
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    save_figure(fig, out_dir, "03_role_skill_heatmap")


def plot_state_bar_chart(
    df: pd.DataFrame, out_dir: Path, top_n: int = 10
) -> None:
    chart_df = (
        df.sort_values(
            ["cloud_any_share", "state_total_postings"],
            ascending=[False, False],
        )
        .head(top_n)
        .sort_values("cloud_any_share", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(chart_df["state"], chart_df["cloud_any_share"])
    ax.set_title("Top States by Cloud/Platform Skill Prevalence")
    ax.set_xlabel("Share of postings mentioning at least one cloud/platform skill")
    ax.set_ylabel("State")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    max_val = chart_df["cloud_any_share"].max()
    ax.set_xlim(0, max_val + 0.18)

    for i, (_, row) in enumerate(chart_df.iterrows()):
        ax.text(
            row["cloud_any_share"] + 0.005,
            i,
            f'{row["cloud_any_share"]:.1%} (n={int(row["state_total_postings"])})',
            va="center",
            fontsize=9,
        )

    save_figure(fig, out_dir, "04_state_cloud_prevalence")

def plot_seniority_grouped_bars(
    df: pd.DataFrame,
    out_dir: Path,
    selected_skills: List[str],
) -> None:
    chart_df = df[df["skill"].isin(selected_skills)].copy()
    chart_df = chart_df[chart_df["seniority_group"].isin(SENIORITY_ORDER)].copy()

    pivot = chart_df.pivot(
        index="seniority_group",
        columns="skill",
        values="posting_share",
    ).fillna(0)

    pivot = pivot.reindex([group for group in SENIORITY_ORDER if group in pivot.index])

    seniority_counts = (
        chart_df.groupby("seniority_group", as_index=False)["seniority_total_postings"]
        .max()
        .set_index("seniority_group")["seniority_total_postings"]
        .to_dict()
    )

    low_n_threshold = 10

    xtick_labels = []
    for group in pivot.index:
        n = int(seniority_counts.get(group, 0))
        marker = "*" if n < low_n_threshold else ""
        xtick_labels.append(f"{group}{marker}\n(n={n})")

    x = np.arange(len(pivot.index))
    n_skills = len(pivot.columns)
    width = 0.8 / n_skills

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, skill in enumerate(pivot.columns):
        ax.bar(
            x + (i - (n_skills - 1) / 2) * width,
            pivot[skill].values,
            width=width,
            label=skill,
        )

    for idx, group in enumerate(pivot.index):
        n = int(seniority_counts.get(group, 0))
        if n < low_n_threshold:
            ax.axvspan(idx - 0.45, idx + 0.45, color="gray", alpha=0.08, zorder=0)

        # x in data coords, y in axes coords
            trans = blended_transform_factory(ax.transData, ax.transAxes)

            ax.text(
                idx,                # center of the grey band
                0.88,               # move up/down manually
                "* Very small sample",
                transform=trans,
                ha="center",
                va="top",
                fontsize=8,
                bbox=dict(
                   boxstyle="round,pad=0.25",
                   facecolor="white",
                 alpha=0.75,
                 edgecolor="none",
                ),
                zorder=5,
            )

    ax.set_title("Skill Demand by Seniority")
    ax.set_xlabel("Seniority group")
    ax.set_ylabel("Share of postings")
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()

    save_figure(fig, out_dir, "05_seniority_skill_bars")

def plot_skill_cooccurrence_heatmap(
    pairs_df: pd.DataFrame,
    top_skills_df: pd.DataFrame,
    out_dir: Path,
    top_n: int = 8,
    exclude_skills: tuple[str, ...] = ("Python",),
) -> None:
    selected_skills = [
        "PyTorch",
        "TensorFlow",
        "LLM",
        "NLP",
        "Generative AI",
        "AWS",
        "Spark",
        "SQL",
    ]
    posting_count_map = dict(zip(top_skills_df["skill"], top_skills_df["posting_count"]))

    filtered_pairs = pairs_df[
        pairs_df["skill_a"].isin(selected_skills) & pairs_df["skill_b"].isin(selected_skills)
    ].copy()

    matrix = pd.DataFrame(
        0.0,
        index=selected_skills,
        columns=selected_skills,
    )

    for _, row in filtered_pairs.iterrows():
        a = row["skill_a"]
        b = row["skill_b"]
        pair_count = row["pair_count"]

        count_a = posting_count_map.get(a, 0)
        count_b = posting_count_map.get(b, 0)

        denom = count_a + count_b - pair_count
        jaccard = pair_count / denom if denom else 0.0

        matrix.loc[a, b] = jaccard
        matrix.loc[b, a] = jaccard

    # Blank diagonal for readability
    np.fill_diagonal(matrix.values, np.nan)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix.values, aspect="auto")

    ax.set_title("Skill Co-occurrence Heatmap (Top Non-Python Skills)")
    ax.set_xlabel("Skill")
    ax.set_ylabel("Skill")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                text_color = "white" if val >= 0.25 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    cbar = fig.colorbar(im, ax=ax, label="Jaccard similarity")
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    save_figure(fig, out_dir, "06_skill_cooccurrence_heatmap")


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    figures_dir = ensure_dir(args.figures_dir)

    top_skills_df = load_csv(analysis_dir, "top_skills.csv")
    monthly_trends_df = load_csv(analysis_dir, "monthly_skill_trends.csv")
    role_matrix_df = load_csv(analysis_dir, "role_skill_matrix.csv")
    state_cloud_df = load_csv(analysis_dir, "state_cloud_summary.csv")
    seniority_matrix_df = load_csv(analysis_dir, "seniority_skill_matrix.csv")
    pairs_df = load_csv(analysis_dir, "skill_pairs.csv")

    plot_top_skills(top_skills_df, figures_dir, top_n=args.top_n)
    plot_skill_timeline(
        monthly_trends_df,
        figures_dir,
        selected_skills=DEFAULT_TREND_SKILLS,
        min_monthly_postings=8,
        smoothing_window=2,
    )
    plot_role_heatmap(
        role_matrix_df,
        figures_dir,
        selected_skills=DEFAULT_HEATMAP_SKILLS,
    )
    plot_state_bar_chart(
        state_cloud_df,
        figures_dir,
        top_n=10,
    )
    plot_seniority_grouped_bars(
    seniority_matrix_df,
    figures_dir,
    selected_skills=DEFAULT_SENIORITY_SKILLS,
    )
    plot_skill_cooccurrence_heatmap(
        pairs_df,
        top_skills_df,
        figures_dir,
        top_n=8,
        exclude_skills=("Python",),
    )

    print(f"Saved figures to {figures_dir.resolve()}")


if __name__ == "__main__":
    main()