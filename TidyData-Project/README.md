# Federal R&D Budget Analysis (1976–2017)

## Overview
This project tidies and analyzes a dataset of U.S. federal R&D budget allocations across
14 government departments from 1976 to 2017, alongside annual GDP figures.

## Dataset
- **File:** `fed_rd_year&gdp.csv`
- **Rows:** 588 | **Departments:** 14 | **Years:** 1976–2017
- **Departments:** DHS, DOC, DOD, DOE, DOT, EPA, HHS, Interior, NASA, NIH, NSF, Other, USDA, VA

## Project Goals
The raw data was in wide format, each year had its own column. The goal was to tidy the dataset
by reshaping it into long format with clean, analysis-ready columns: `department`, `year`, `rd_budget`, and `gdp`.

## Steps
1. **Import Libraries & Load Data** — Load and inspect the raw dataset
2. **Melt the DataFrame** — Reshape from wide to long format using `pd.melt()`
3. **Split Year and GDP Columns** — Separate combined `year_gdp` values into two clean columns
4. **Handle Missing Values & Data Types** — Fill NaNs with 0, convert `gdp` to numeric
5. **Descriptive Statistics** — Summary of the cleaned dataset
6. **Visualizations** — R&D budget over time, GDP over time, budget by department

## Key Findings
- **DOD** has consistently received the largest share of federal R&D funding
- **HHS and NIH** showed the greatest growth over the 40-year period (~$25B each)
- **NASA, DOT, EPA, and Interior** saw budget declines by 2017
- Overall federal R&D spending and U.S. GDP both trended upward from 1976 to 2017

## Requirements
```
pandas
numpy
seaborn
matplotlib
```
