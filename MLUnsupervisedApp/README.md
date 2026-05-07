# Learning Without Supervision: Interactive Unsupervised ML App

## Overview

This project is an interactive Streamlit web app that teaches users three core unsupervised machine
learning algorithms — **Principal Component Analysis (PCA)**, **K-Means Clustering**, and
**Hierarchical Clustering** — through hands-on exploration, no coding required.

**Unsupervised Learning:** A type of machine learning where the model discovers structure in data
*without any labels or correct answers*. Unlike supervised learning, the algorithm is given only raw
features and must find patterns, groupings, or compressed representations on its own.

Users can work with a built-in sample dataset or upload their own CSV to see how each algorithm
behaves on real data — adjusting hyperparameters and watching results update in real time.

## Live Demo

[Launch the app](https://ml-unsupervised-app.streamlit.app/)

## How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/eromero6/ROMERO-Data-Science-Portfolio.git
cd ROMERO-Data-Science-Portfolio/MLUnsupervisedApp
```

2. Install dependencies (see Requirements below):
```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install streamlit pandas numpy matplotlib scikit-learn scipy plotly
```

3. Run the app:
```bash
streamlit run final_project.py
```

4. The app opens automatically at `http://localhost:8501`

## Requirements

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | 1.53.0 | Web app framework and interactive UI |
| `pandas` | 2.3.3 | Data loading and manipulation |
| `numpy` | 2.4.1 | Numerical operations and array handling |
| `matplotlib` | 3.10.8 | Static plot rendering (dendrograms, scatter plots, scree plots) |
| `scikit-learn` | 1.8.0 | PCA, K-Means, Agglomerative Clustering, preprocessing, and metrics |
| `scipy` | 1.17.1 | Hierarchical linkage matrix and dendrogram computation |
| `plotly` | 6.6.0 | Interactive scatter plots and world map choropleth |

## App Features

### Navigation

The home page presents three model cards. Clicking any card navigates to that model's dedicated page.
Each page supports two data modes: **Sample Dataset** (pre-loaded) and **Upload CSV File** (your own data).

---

### 1. Principal Component Analysis (PCA)

**What it does:** Reduces high-dimensional data to fewer dimensions (principal components) while
retaining as much variance as possible. Used for visualization, noise reduction, and preprocessing.

**Sample dataset:** Breast Cancer Wisconsin — 569 tumor biopsy samples, 30 numeric features,
binary labels (malignant / benign).

**Hyperparameters & controls:**
| Control | What it does |
|---|---|
| Scree Plot slider (1–15) | Selects how many principal components to retain; updates variance explained % and moves the red cut-line on the scree plot |

**Visualizations:**
- **Scatter Plot** — 2D projection onto PC1 and PC2, colored by true labels
- **PCA Loadings** — horizontal bar chart showing each feature's contribution to PC1 and PC2
- **Scree Plot** — cumulative explained variance curve with an interactive vertical cut-line

**Metrics shown:** PC1 variance %, PC2 variance %, combined variance %, variance explained by chosen number of components

---

### 2. K-Means Clustering

**What it does:** Partitions data into *k* groups by iteratively assigning points to the nearest
centroid and updating centroids until convergence. The number of clusters *k* must be specified upfront.

**Sample dataset:** Breast Cancer Wisconsin (same as PCA).

**Hyperparameters & controls:**
| Control | What it does |
|---|---|
| k slider (2–10) | Sets the number of clusters; all metrics and plots update immediately |

**Visualizations:**
- **Cluster Scatter Plot** — 2D PCA projection colored by cluster assignment
- **True Labels Comparison** — side-by-side scatter of K-Means clusters vs. known diagnoses, with accuracy score (k = 2 only)
- **Elbow & Silhouette** — WCSS elbow curve and silhouette score curve across k = 2–10 to guide k selection

**Metrics shown:** Clusters (k), Inertia (WCSS), Silhouette Score, Accuracy Score (k = 2 only)

> **Note on accuracy:** Because K-Means assigns cluster numbers arbitrarily, both possible label
> orderings are checked and the higher accuracy is reported.

---

### 3. Hierarchical (Agglomerative) Clustering

**What it does:** Builds a tree of nested clusters (a *dendrogram*) by progressively merging the
most similar observations. The number of clusters is chosen by "cutting" the tree at a chosen height,
so you can explore different granularities without re-fitting.

**Sample dataset:** Democracy & Dictatorship 2020 — 184 countries described by 17 political and
electoral indicators (sourced from [TidyTuesday](https://github.com/rfordatascience/tidytuesday/tree/main/data/2024/2024-11-05)).

**Hyperparameters & controls:**
| Control | What it does |
|---|---|
| Linkage method (ward / complete / average / single) | Controls how inter-cluster distance is measured; changes cluster shape and the dendrogram structure |
| k slider (2–8) | Sets the cut height on the tree; red dashed line on the dendrogram updates in real time |

**Visualizations:**
- **Dendrogram** — truncated tree (last 30 merges) with a red dashed cut-line marking your chosen k
- **Cluster Scatter Plot (PCA)** — interactive Plotly scatter with country hover labels
- **World Map** — Plotly choropleth showing cluster assignments by country (sample dataset only)
- **Silhouette Analysis** — silhouette scores across k = 2–8 for the selected linkage method

**Metrics shown:** Clusters (k), Silhouette Score, Countries/Rows Clustered, Cluster sizes

---

### Upload CSV (all three models)

All three models accept a user-uploaded CSV file. The app:
1. Auto-detects numeric columns to use as features
2. Lets users select/deselect columns via a multiselect widget (Hierarchical) or drops non-numeric columns automatically
3. Drops rows with missing values and reports how many rows remain
4. Standardizes features before fitting (StandardScaler — mean = 0, std = 1)

**Guidelines for upload:**
- Feature columns must be numeric (integers or decimals)
- Remove ID or index columns before uploading — these distort distance calculations
- Keep any label/target column if you want to color cluster plots or compare against true groups

---

## Pre-processing Steps

All three models follow the same standardization pipeline:

1. Select numeric feature columns
2. Drop rows with any missing values in those columns
3. Apply `StandardScaler` (zero mean, unit variance) — required because PCA, K-Means, and
   hierarchical clustering all depend on distance or variance, which are scale-sensitive

---

## References

- [Scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Scikit-learn: KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Scikit-learn: AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [Scipy: Hierarchical clustering (linkage, dendrogram)](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [Breast Cancer Wisconsin Dataset — UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- [Democracy & Dictatorship Dataset — TidyTuesday 2024](https://github.com/rfordatascience/tidytuesday/tree/main/data/2024/2024-11-05)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)
- [Understanding Silhouette Score](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)
- [Elbow Method for K-Means](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/)
