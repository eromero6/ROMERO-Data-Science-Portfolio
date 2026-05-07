# Data Science Portfolio

A collection of data science projects built with Python, covering data wrangling, interactive web applications, supervised machine learning, and unsupervised machine learning. Projects progress from foundational EDA to deployed, user-facing ML tools.

Please refer to the table of contents for a quick explanation of each project and the relevant skills used.

---

## Table of Contents

| Project | Description | Skills |
|---|---|---|
| [Tidy Data Project](TidyData-Project/) | Tidying and analyzing U.S. federal R&D budget data (1976–2017) | Data wrangling, pandas, seaborn, matplotlib |
| [Streamlit App](basic_streamlit_app/) | Interactive data web application — [Live Demo](https://mainpy-bitvndnj83rocsasqj9oc6.streamlit.app/) | Python, Streamlit |
| [ML Streamlit App](MLStreamlitApp/) | Interactive supervised learning app with model training and evaluation — [Live Demo](https://romero-data-science-portfolio-fwkdgwtp9gyexbkbgwjduj.streamlit.app/) | scikit-learn, Streamlit, classification, data preprocessing |
| [ML Unsupervised App](MLUnsupervisedApp/) | Interactive unsupervised learning app covering PCA, K-Means, and Hierarchical Clustering | scikit-learn, scipy, Plotly, Streamlit, dimensionality reduction, clustering |

---

## Projects

### Tidy Data Project
**[View Project](TidyData-Project/)**

An analysis of U.S. federal R&D budget allocations across 14 government departments from 1976 to 2017.
The raw dataset was in wide format with each year as its own column. Using `pd.melt()` and string splitting,
the data was reshaped into a clean, tidy format following Hadley Wickham's tidy data principles.
Visualizations were created to explore spending trends over time and compare budgets across departments.

**How it complements this portfolio:** This project demonstrates foundational data wrangling skills, specifically the ability to take messy, real-world data and transform it into a structured format ready for analysis.
These skills are essential to every project in this portfolio, as clean data is the foundation of
any meaningful machine learning or visualization work.

### Streamlit App: Oasis Discography
**[View Project](basic_streamlit_app/)**

An interactive web app built with Streamlit that explores the audio features of every Oasis song and album.
The app uses Spotify audio data to visualize characteristics like danceability, energy, acousticness,
loudness, and speechiness across the band's full discography. Users can filter by album and explore
track durations interactively.

**How it complements this portfolio:** This project demonstrates the ability to build and deploy
interactive data applications, going beyond static analysis to create a shareable, user-facing product.
It bridges data science and web development, a key skill for communicating insights to non-technical audiences.

### ML Streamlit App: Teaching Machines
**[View Project](MLStreamlitApp/)**

An interactive supervised machine learning app built with Streamlit. Users can load one of four
built-in datasets (Iris, Penguins, Titanic, Tips) or upload their own CSV, select a target variable
and features, tune model hyperparameters, and train a Logistic Regression or Decision Tree classifier.
Results are displayed across two tabs — Model Results (accuracy, ROC AUC, classification report,
coefficients or feature importances) and Visualizations (confusion matrix, predicted probabilities,
ROC curve, and an interactive decision tree diagram). Every metric and chart includes a plain-language
explanation to make the app accessible to learners at any level.

**How it complements this portfolio:** This project brings together data preprocessing, machine learning,
and interactive application design in one end-to-end tool. It demonstrates the ability to translate
model outputs into understandable insights — a critical skill for communicating data science work
to non-technical audiences.

### ML Unsupervised App: Learning Without Supervision
**[View Project](MLUnsupervisedApp/)**

An interactive unsupervised machine learning app built with Streamlit. Users can explore three
algorithms — Principal Component Analysis (PCA), K-Means Clustering, and Hierarchical Clustering —
using a built-in sample dataset or their own uploaded CSV. Each model page walks through data
standardization, hyperparameter tuning via sliders and dropdowns, and multiple interactive
visualizations. PCA includes a scree plot with a movable component selector and a loadings chart.
K-Means features an elbow and silhouette plot to guide the choice of k, with an accuracy comparison
against true labels. Hierarchical Clustering always renders the dendrogram first so users can
visually identify a natural cut before selecting k, with an interactive world map choropleth
(Plotly) for the sample dataset. Every section closes with a key takeaways panel that evaluates
results in plain language.

**How it complements this portfolio:** This project extends the portfolio into unsupervised learning —
a fundamentally different paradigm where the algorithm finds structure without labels. It demonstrates
the ability to build multi-model educational tools that guide users through algorithm intuition,
preprocessing decisions, hyperparameter exploration, and result interpretation in a single, cohesive interface.

---

*This repository was created by Eva S. Romero*
