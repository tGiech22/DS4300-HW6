# Job Market Skill Demand Analyzer

## 1. Project Framing

### Working Title
Skill Demand in U.S. AI and Machine Learning Job Postings

### Final Project Claim
This project uses MongoDB and Python-based text analysis to identify the technical skills most frequently requested in U.S. AI and machine learning job postings and to measure how those demands vary across geography, seniority, job role, and time.

### Why This Fits the Assignment
- **NoSQL justification:** job postings are semi-structured and text-heavy, with inconsistent location, title, and description fields
- **Scientific/cultural/economic significance:** labor-market demand for AI and data skills affects hiring, education, workforce preparation, and regional economic competitiveness
- **Data science investigation:** the project includes data cleaning, text mining, aggregation, trend analysis, and visualization
- **Originality:** the focus is on skill bundles and demand patterns within AI/ML jobs, not sports, games, or a Kaggle-only toy analysis

## 2. Dataset Choice

### Primary Dataset
[`Job_Postings_US new.csv`](/Users/van/DS4300-HW6/Job_Postings_US%20new.csv)

### Why This Dataset Works
- Contains **997 real-looking job postings**
- Date coverage spans **December 20, 2022 to April 9, 2025**
- Includes **full job descriptions**, which is the critical field needed for skill extraction
- Includes **job title**, **company**, **city/state**, and **seniority**
- Supports a real NoSQL use case because each posting can be enriched with extracted fields without forcing a rigid relational schema

### Why `future_jobs_dataset.csv` Is Weaker
- Already contains a pre-filled `skills_required` field
- Appears highly structured and likely synthetic
- Does not support a strong NoSQL justification
- Removes the most interesting part of the project: extracting skills from messy posting text

## 3. Concrete Research Questions

Use these as the backbone of the report:

1. Which technical skills appear most often in U.S. AI and machine learning job postings?
2. How does skill demand differ across job roles such as `Machine Learning Engineer`, `Data Scientist`, and `Software Engineer, Machine Learning`?
3. How does skill demand vary by seniority level?
4. Which skills or skill bundles become more common over time?
5. Which regions show stronger demand for cloud, programming, or deep-learning tools?

These questions are measurable with the available columns.

## 4. Project Scope

### Recommended Filtering Rules
- Keep roles clearly related to AI, ML, and data science
- Exclude obvious sports/game-centered postings if they distort the assignment scope
- Standardize location fields such as `California` and `CA`
- Consolidate title variants into broader role groups

### Recommended Role Groups
- `Machine Learning Engineer`
- `Data Scientist`
- `ML Software Engineer`
- `Applied Scientist`
- `AI Engineer`
- `Other AI/Data`

### Recommended Seniority Groups
- `Internship`
- `Entry level`
- `Associate`
- `Mid-Senior level`
- `Director/Executive`
- `Unknown/Other`

## 5. Why MongoDB Is the Right NoSQL Choice

MongoDB is the most practical option because each posting can be stored as a document and incrementally enriched during the project.

### Raw Posting Document
```json
{
  "_id": 123,
  "job_posted_date": "2025-03-14",
  "company_name": "Ikigai",
  "company_address_locality": "San Francisco",
  "company_address_region": "California",
  "company_description": "...",
  "job_title": "AI/ML Engineer",
  "seniority_level": "Mid-Senior level",
  "job_description_text": "..."
}
```

### Enriched Posting Document
```json
{
  "_id": 123,
  "source_file": "Job_Postings_US new.csv",
  "raw": {
    "job_posted_date": "2025-03-14",
    "company_name": "Ikigai",
    "company_address_locality": "San Francisco",
    "company_address_region": "California",
    "company_description": "...",
    "job_title": "AI/ML Engineer",
    "seniority_level": "Mid-Senior level",
    "job_description_text": "..."
  },
  "normalized": {
    "posted_month": "2025-03",
    "state": "CA",
    "city": "San Francisco",
    "role_group": "Machine Learning Engineer",
    "seniority_group": "Mid-Senior level"
  },
  "skills": {
    "programming": ["Python", "SQL"],
    "ml_frameworks": ["PyTorch", "TensorFlow"],
    "cloud": ["AWS"],
    "data_tools": ["Spark", "Pandas"],
    "visualization": [],
    "all": ["Python", "SQL", "PyTorch", "TensorFlow", "AWS", "Spark", "Pandas"]
  }
}
```

### Why This Matters
- New extracted fields can be added without schema migration
- Skills can be nested by category
- Aggregations on title, state, date, and skills are straightforward
- Raw and cleaned representations can coexist in one document

## 6. Data Pipeline

### Step 1: Load the CSV
Read the file with Python and insert each row into a MongoDB `raw_postings` collection.

### Step 2: Clean and Normalize
- Parse `job_posted_date`
- Normalize `company_address_region` into state abbreviations
- Strip boilerplate and whitespace from description fields
- Map noisy titles into broader role groups
- Normalize seniority labels

### Step 3: Extract Skills
Build a controlled vocabulary of skills and scan each description for exact and variant matches.

### Step 4: Store Enriched Results
Write cleaned documents to `clean_postings` or append normalized fields to the same collection.

### Step 5: Analyze
Run MongoDB aggregation pipelines or pandas analysis over exported query results.

## 7. Skill Extraction Method

Use a dictionary-based method. It is transparent, defensible, and easy to explain in a report.

### Suggested Skill Categories
- **Programming:** Python, R, SQL, Java, Scala, C++, Rust
- **ML/AI frameworks:** PyTorch, TensorFlow, Keras, Scikit-learn, XGBoost
- **Data engineering:** Spark, Hadoop, Airflow, Kafka, Databricks
- **Cloud/platforms:** AWS, Azure, GCP, Kubernetes, Docker
- **Databases/search:** PostgreSQL, MySQL, MongoDB, Elasticsearch, DynamoDB
- **Analytics/BI:** Tableau, Power BI, Excel
- **LLM/GenAI:** NLP, LLM, LangChain, transformers, prompt engineering

### Important Normalization Rules
- Map `postgres` to `PostgreSQL`
- Map `gcp` to `GCP`
- Treat `scikit learn` and `sklearn` as `Scikit-learn`
- Handle ambiguous tokens carefully, especially `R`

### Method Statement for Report
“Skills were extracted using a rule-based dictionary of technical terms and common variants. The method prioritizes interpretability and consistent measurement across postings over opaque model-based extraction.”

## 8. Analysis Plan

### Core Metrics
- Posting count by month
- Frequency of each skill
- Share of postings containing each skill
- Skill frequency by role group
- Skill frequency by seniority
- Skill frequency by state
- Skill co-occurrence counts

### Strong Analytical Angles
- Compare `Python`, `SQL`, and cloud-tool prevalence across roles
- Compare deep-learning stack demand in `Machine Learning Engineer` vs `Data Scientist`
- Measure whether `LLM` and `GenAI` terms increase in later postings
- Identify skill bundles that commonly appear together

## 9. Required Visualizations

Use at least four. Six would be safer for a 5+ page report.

### Visualization 1: Top Skills Overall
- Horizontal bar chart of the 15 most frequent skills
- Purpose: establish baseline demand

### Visualization 2: Skill Demand Over Time
- Monthly line chart for selected skills such as `Python`, `SQL`, `AWS`, `PyTorch`, `LLM`
- Purpose: show which tools are stable versus emerging

### Visualization 3: Skill Demand by Role Group
- Heatmap of skills by role group
- Purpose: show that different AI roles require different tool stacks

### Visualization 4: Skill Demand by Seniority
- Stacked or grouped bar chart
- Purpose: show whether advanced roles emphasize cloud/distributed systems more heavily than internships or entry-level roles

### Visualization 5: Geography
- Bar chart of top states with cloud/deep-learning skill prevalence
- Purpose: connect skill demand to regional hiring patterns

### Visualization 6: Skill Co-occurrence Network
- Network graph or co-occurrence matrix
- Purpose: reveal common skill bundles, such as `Python + SQL + AWS`

## 10. Likely Conclusions You Can Defend

Do not pre-write conclusions as facts, but this dataset is likely to support conclusions of this form:

- `Python` is the most universal skill across AI/ML roles
- `SQL` remains essential even in machine-learning-heavy roles
- `PyTorch` and `TensorFlow` appear more often in engineering and deep-learning roles than in general data-science titles
- Cloud and platform tools such as `AWS`, `Kubernetes`, and `Docker` are more common in senior engineering roles
- Demand for `LLM` and generative-AI language likely increases in more recent postings

These are hypotheses to test, not assumptions to hard-code.

## 11. Limitations Section

You should explicitly acknowledge:
- The dataset contains only 997 postings, so some trend claims should be cautious
- The postings are not a random sample of the entire U.S. labor market
- Some postings may come from repeated platforms or duplicate role families
- Dictionary-based extraction may miss implicit or unusual skill mentions
- State labels are inconsistent and require normalization

Including limitations will make the report stronger.

## 12. Report Structure

This structure should comfortably reach 5+ pages.

### 1. Introduction
- Why AI/ML labor-market skill demand matters
- Project goal and research questions

### 2. Data Source and NoSQL Justification
- Describe the CSV fields
- Explain why MongoDB fits semi-structured postings

### 3. Methods
- Data cleaning
- Title grouping
- State normalization
- Skill dictionary creation
- MongoDB schema
- Aggregation and plotting workflow

### 4. Results
- Present each visualization
- Interpret the findings directly under each chart

### 5. Discussion
- Explain what the findings mean for job seekers, educators, or firms
- Discuss trends and regional differences

### 6. Limitations and Future Work
- Sample bias
- Extraction limitations
- Potential improvements such as more data sources or named-entity extraction

### 7. Conclusion
- Answer the research question clearly
- State the strongest 2 to 3 findings

## 13. Code Deliverables

Minimum recommended code files:

- `src/load_to_mongo.py`
- `src/clean_postings.py`
- `src/extract_skills.py`
- `src/analyze_skills.py`
- `notebooks/final_analysis.ipynb`

## 14. Practical Execution Plan

### Phase 1: Data and Database
- Load the CSV into MongoDB
- Verify document counts
- Inspect missing and inconsistent fields

### Phase 2: Cleaning and Enrichment
- Normalize states, dates, and titles
- Create skill dictionary
- Extract and store skill arrays

### Phase 3: Analysis and Visuals
- Compute frequencies and trends
- Generate plots
- Save final figures for the report

### Phase 4: Report Writing
- Insert figures
- Explain methods
- Write conclusions and limitations

## 15. Best Final Title Options

- Skill Demand in U.S. AI and Machine Learning Job Postings
- What Employers Want: A NoSQL Analysis of AI and ML Job Skills
- Tracking AI Labor-Market Demand Through Semi-Structured Job Postings
