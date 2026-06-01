# Chicago E-Scooter Trips: Exploratory Data Analysis

An end-to-end exploratory data analysis of over 2.2 million e-scooter trips taken in Chicago during 2025, using data from the City of Chicago Data Portal. This project investigates when, where, and how Chicagoans ride shared e-scooters — uncovering commute patterns, geographic disparities, vendor dynamics, and the (surprisingly weak) relationship between trip distance and duration.

---

## Dataset

**Source:** [City of Chicago Data Portal — E-Scooter Trips](https://data.cityofchicago.org/d/2i5w-ykuw)

**Scope:** January 2025 – present (filtered from the full 12M+ row historical dataset)

| Field | Description |
|---|---|
| Trip ID | Unique identifier per trip |
| Start / End Time | Datetime of trip start and end |
| Trip Distance | Distance traveled (converted from meters to miles) |
| Trip Duration | Duration in minutes (converted from seconds) |
| Vendor | Lime or Lyft |
| Start / End Community Area | Chicago neighborhood names |

**Final dataset:** 2,290,110 rows × 16 columns after cleaning and feature engineering.

---

## Getting the Data

The dataset is not included in this repo (677 MB exceeds GitHub's file limit). To run the notebook locally:

1. Go to the [City of Chicago Data Portal — E-Scooter Trips](https://data.cityofchicago.org/d/2i5w-ykuw)
2. Filter the export to your desired date range (this analysis uses January 2025 onward)
3. Export as CSV and save it as `scooter_data.csv` in this folder

The notebook expects the raw portal export — all unit conversions and cleaning are performed inside.

---

## Project Structure

```
EOC Portfolio Update/
├── chicago_scooter_project.ipynb   # Main analysis notebook
├── scooter_data.csv                # Not included — see "Getting the Data" above
└── README.md
```

---

## Analysis Overview

### 1. Data Cleaning & Preparation
- Converted units: meters → miles, seconds → minutes
- Parsed datetime strings and extracted hour, date, and day-of-week features
- Engineered a binary `Weekend` indicator
- Identified and documented extreme outliers (trips flagged at 2,000+ miles — likely GPS errors)
- Verified 0 duplicate rows; retained 4,366 rows with missing community area names (0.19% — negligible)

### 2. Summary Statistics & Distributions
- Trip distance: mean 1.20 mi, median 0.83 mi (right-skewed)
- Trip duration: mean 10.0 min, median 6.5 min (right-skewed)
- Vendor split: **Lime 82.9%** vs. Lyft 17.1%

### 3. Usage Patterns
- **By hour:** Dual peaks on weekdays (8–9 AM and 3–6 PM); single gradual build on weekends
- **By day:** Friday 5 PM is the single busiest slot (~35,284 trips); late-night spikes on Friday/Saturday
- **By neighborhood:** Lake View, Lincoln Park, Near North Side, West Town, and Near West Side dominate; deep South Side neighborhoods show significantly lower ridership

### 4. Hypothesis Testing

| Test | Result |
|---|---|
| Weekday vs. weekend trip counts (two-sample t-test) | t = 0.605, p = 0.546 — **fail to reject H₀** (no significant volume difference) |
| Correlation: trip duration ~ distance (Pearson) | r = 0.087, p < 0.0001 — **reject H₀**, but effect is weak |

---

## Key Findings

- **Commute dominates weekday usage** — morning and evening peaks align with standard Chicago work commutes.
- **Weekends shift toward leisure** — trips are longer and more spread across the day; lakefront neighborhoods (Edgewater, Uptown, Lake View) see notably higher Saturday ridership, consistent with recreational riding along the lakefront path.
- **Trip duration is a poor predictor of distance** (r = 0.087) — factors like traffic, construction, and route choice likely explain more variance.
- **Despite different usage patterns, total daily trip volume is statistically similar on weekdays vs. weekends** — Chicago's diverse population means riders on different schedules roughly balance out.
- **Geographic equity gap** — high-usage corridors cluster on the North Side and downtown; low usage on the South Side may reflect infrastructure gaps (bike lanes), socioeconomic factors, or vendor deployment decisions.
- **Lime holds an overwhelming market share (83%)** with minimal geographic variation except a slight uptick in Lyft usage in the Loop.

---

## Visualizations

- Heatmap of trip volume by day of week × hour of day
- KDE plot comparing weekday vs. weekend trip duration distributions
- Clustermap of top 20 neighborhoods by day of week
- Facet grid of hourly usage patterns across days
- Scatter plot of trip distance vs. duration

---

## Technologies

| Tool | Use |
|---|---|
| Python 3 | Core language |
| pandas | Data loading, cleaning, feature engineering |
| matplotlib / seaborn | Static visualizations |
| scipy | Hypothesis testing (t-test, Pearson correlation) |
| Jupyter Notebook | Interactive analysis environment |

---

## Limitations

- Analysis covers **2025 only** — the full dataset spans 2019–present but was too large to extract locally (12M+ rows)
- Outlier trips (2,000+ mile distances) indicate GPS or data entry errors; they were documented but not removed, to preserve data integrity
- No weather, temperature, or event data incorporated — these likely explain significant variance in daily ridership
- Cannot assess year-over-year trends or COVID-era shifts with a single-year snapshot

---

## Potential Next Steps

- Incorporate multi-year data to analyze adoption trends, seasonality, and COVID-era impacts
- Add weather data to model ridership as a function of temperature and precipitation
- Build a time-series or regression model to forecast daily trip counts
- Investigate the geographic equity gap more formally — overlay census income/demographics data with ridership by neighborhood
- Analyze trip-level routes using the latitude/longitude coordinates (dropped in this analysis)
