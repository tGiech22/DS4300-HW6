# Assignment: DS4300 Homework 6
# Written by: Gavin Bond & Erika Sohn
# Function: Generate insightful visuals from skill analysis

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List 

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter   
from matplotlib.transforms import blended_transform_factory  
import numpy as np                               
import pandas as pd                              

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

DEFAULT_SENIORITY_SKILLS = ["SQL", "AWS", "Docker", "PyTorch", "LLM"]

ROLE_ORDER = [
    "Machine Learning Engineer",
    "Data Scientist",
    "ML Software Engineer",
    "Other AI/Data",
    "ML Platform / MLOps / Infrastructure",
    "Data / ML Engineer",
    "Applied Scientist / Research",
]

CATEGORY_COLORS = {
    "programming":      "#4C72B0",
    "ml_frameworks":    "#DD8452",
    "data_engineering": "#55A868",
    "cloud_platforms":  "#C44E52",
    "databases":        "#8172B2",
    "analytics_bi":     "#937860",
    "llm_genai":        "#DA8BC3",
    "unknown":          "#CCCCCC",
}

PMI_SKILLS = [
    "PyTorch", "TensorFlow", "LLM", "NLP",
    "Generative AI", "AWS", "Spark", "SQL",
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


def plot_top_skills(
        top_skills_df: pd.DataFrame, 
        top_skills_cat_df: pd.DataFrame, 
        out_dir: Path, 
        top_n: int = 15
) -> None:
    
    # zip "skill" and "categories" columns into a dict 
    skill_to_cat = dict(zip(top_skills_cat_df["skill"], top_skills_cat_df["category"]))

    # build skill -> category lookup from top_skills_by_category.csv
    
    chart_df = (
        top_skills_df.sort_values(["posting_share", "posting_count"], ascending=[False, False])
        .head(top_n)
        .sort_values("posting_share", ascending=True)
    )

    colors = [
        CATEGORY_COLORS.get(skill_to_cat.get(skill, "unknown"), CATEGORY_COLORS["unknown"])
        for skill in chart_df["skill"]
    ]

    # for every skill, apply color respective to its category
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(chart_df["skill"], chart_df["posting_share"], color=colors)
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
    
    present_cats = set(skill_to_cat.get(s, "unknown") for s in chart_df["skill"])
    legend_patches = [
        mpatches.Patch(color=CATEGORY_COLORS[cat], label=cat.replace("_", " ").title())
        for cat in CATEGORY_COLORS if cat in present_cats
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, title="Category")
    save_figure(fig, out_dir, "01_top_skills_by_category")

def plot_tfidf(
    tfidf_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    
    roles = tfidf_df["role"].unique()
    n_roles = len(roles)
    ncols = 2
    nrows = (n_roles + 1) // ncols 
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = axes.flatten()
    
    for i, role in enumerate(roles):
        role_df = tfidf_df[tfidf_df["role"] == role].sort_values("score", ascending=True)
        axes[i].barh(role_df["term"], role_df["score"], color="#4C72B0")
        axes[i].set_title(role, fontsize=9, fontweight="bold")
        axes[i].set_xlabel("TF-IDF score", fontsize=8)
        axes[i].tick_params(axis="y", labelsize=8)

    # hide unused subplots for odd number of roles 
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distinctive Terms by Role Group (TF-IDF)", fontsize=13, fontweight="bold")
    save_figure(fig, out_dir, "02_tfidf_by_role")

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

def plot_skill_cooccurrence_pmi(
    pairs_df: pd.DataFrame,
    top_skills_df: pd.DataFrame,
    out_dir: Path,
    top_n: int = 8,
    exclude_skills: tuple[str, ...] = ("Python",),
) -> None:
    total = top_skills_df["posting_count"].sum()
    posting_count_map = dict(zip(top_skills_df["skill"], top_skills_df["posting_count"]))

    # create a pair of skill a and skill b
    filtered_pairs = pairs_df[
        pairs_df["skill_a"].isin(PMI_SKILLS) & pairs_df["skill_b"].isin(PMI_SKILLS)
    ].copy()

    # create arr of 0.0 and populate with PMI_SKILLS
    matrix = pd.DataFrame(0.0, index=PMI_SKILLS, columns=PMI_SKILLS)

    # calculate p(a), p(b), and p(a, b)
    for _, row in filtered_pairs.iterrows():
        a, b = row["skill_a"], row["skill_b"]
        p_a = posting_count_map.get(a, 0) / total
        p_b = posting_count_map.get(b, 0) / total
        p_ab = row["pair_count"] / total

        # PMI = log(P (a, b) / (P(a) * P(b)))
        # use np to apply log() 
        pmi = np.log(p_ab / (p_a * p_b))
        matrix.loc[a, b] = round(pmi, 4) # set cell value to pmi score 
        matrix.loc[b, a] = round(pmi, 4)

    np.fill_diagonal(matrix.values, np.nan) # nan diagonal values 

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd") 
    ax.set_title("Skill Co-occurrence Heatmap (PMI)")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                text_color = "white" if val >= 2.0 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    fig.colorbar(im, ax=ax, label="PMI score") 
    save_figure(fig, out_dir, "06_skill_cooccurrence_pmi")
   

def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    figures_dir = ensure_dir(args.figures_dir)

    top_skills_df = load_csv(analysis_dir, "top_skills.csv")
    top_skills_by_cat_df = load_csv(analysis_dir, "top_skills_by_category.csv")  
    tfidf_df = load_csv(analysis_dir, "nlp_tfidf.csv")                           
    role_matrix_df = load_csv(analysis_dir, "role_skill_matrix.csv")
    state_cloud_df = load_csv(analysis_dir, "state_cloud_summary.csv")
    seniority_matrix_df = load_csv(analysis_dir, "seniority_skill_matrix.csv")
    pairs_df = load_csv(analysis_dir, "skill_pairs.csv")

    plot_top_skills(top_skills_df, top_skills_by_cat_df, figures_dir, top_n=args.top_n)  
    plot_tfidf(tfidf_df, figures_dir)                                                      
    plot_role_heatmap(role_matrix_df, figures_dir, selected_skills=DEFAULT_HEATMAP_SKILLS)
    plot_state_bar_chart(state_cloud_df, figures_dir, top_n=10)
    plot_seniority_grouped_bars(seniority_matrix_df, figures_dir, selected_skills=DEFAULT_SENIORITY_SKILLS)
    plot_skill_cooccurrence_pmi(pairs_df, top_skills_df, figures_dir)                    

    print(f"Saved 6 figures to {figures_dir.resolve()}")
    

if __name__ == "__main__":
    main()