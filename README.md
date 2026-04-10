# DS4300-HW6

## Project Overview
This project uses a NoSQL database to study skill demand in U.S. machine learning and data-science job postings. The analysis is built from semi-structured posting text in [`Job_Postings_US new.csv`](/Users/van/DS4300-HW6/Job_Postings_US%20new.csv), which contains job titles, companies, locations, dates, seniority labels, company descriptions, and full job descriptions.

The project is framed as a labor-market investigation rather than a generic dashboard:

**Research question:** Which technical skills are most in demand in U.S. AI and machine learning job postings, and how do those demands vary by role, seniority, geography, and time?

This framing fits the assignment because it:
- uses a NoSQL database for semi-structured text-heavy records
- investigates an economically meaningful question
- documents data sources and methods
- supports multiple insightful visualizations
- leads to defensible conclusions about workforce demand

## Recommended Project Deliverables
- A MongoDB database containing raw and cleaned job posting documents
- Python code for ingestion, cleaning, skill extraction, and analysis
- 4 to 6 visualizations with written interpretation
- A 5+ page report structured around the research question, methods, results, and conclusions

## Repo Files
- [`PROJECT_PLAN.md`](/Users/van/DS4300-HW6/PROJECT_PLAN.md): concrete project specification and report blueprint
- [`requirements.txt`](/Users/van/DS4300-HW6/requirements.txt): suggested Python dependencies
