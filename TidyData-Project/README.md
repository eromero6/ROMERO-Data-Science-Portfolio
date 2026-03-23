# Federal R&D Budget Analysis (1976–2017)

## Overview
This project tidies and analyzes a dataset of U.S. federal R&D budget allocations across
14 government departments from 1976 to 2017, alongside annual GDP figures.

**Tidy Data Principles:** A tidy dataset follows three rules — each variable has its own column,
each observation has its own row, and each value has its own cell. The raw dataset violated these
principles by storing years as column headers, which this project corrects using `pd.melt()`.

## How to Run
1. Clone the repository
2. Install dependencies (see Requirements below)
3. Open `project.ipynb` in Jupyter Notebook or VS Code
4. Run all cells from top to bottom

## Requirements
Install dependencies with:
```bash
pip install pandas numpy seaborn matplotlib
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading, tidying, and aggregation |
| `numpy` | Numerical operations |
| `seaborn` | Statistical visualizations |
| `matplotlib` | Plot formatting |

## Dataset
- **File:** `fed_rd_year&gdp.csv`
- **Source:** U.S. federal government R&D spending records
- **Rows:** 588 | **Departments:** 14 | **Years:** 1976–2017
- **Departments:** DHS, DOC, DOD, DOE, DOT, EPA, HHS, Interior, NASA, NIH, NSF, Other, USDA, VA

### Pre-processing Steps
1. Melted wide format into long format using `pd.melt()`
2. Split combined `year_gdp` column into separate `year` and `gdp` columns
3. Converted `gdp` from object to numeric using `pd.to_numeric()`
4. Filled 26 missing `rd_budget` values with `0` (DHS did not exist before 2002)

## Key Findings
- **DOD** has consistently received the largest share of federal R&D funding
- **HHS and NIH** showed the greatest growth over the 40-year period (~$25B each)
- **NASA, DOT, EPA, and Interior** saw budget declines by 2017
- Overall federal R&D spending and U.S. GDP both trended upward from 1976 to 2017

## References
- [Tidy Data — Hadley Wickham (2014)](https://vita.had.co.nz/papers/tidy-data.pdf)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Seaborn Documentation](https://seaborn.pydata.org)
