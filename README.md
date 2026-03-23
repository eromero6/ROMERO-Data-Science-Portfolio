# Data Science Portfolio

This directory contains the projects I have worked on during my Introduction to Data Science course. Throughout the course, I will learn core machine learning concepts including model selection, feature engineering, hyperparameter tuning, and model evaluation. I will also gain exposure to topics such as neural networks, natural language processing (NLP), and fine-tuning pre-trained LLMs for analytics and decision-making. By the end of this course, I will have created a portfolio of data-driven web applications, integrating exploratory data analysis, machine learning models, and LLMs to solve practice problems and communicate findings.

Please refer to the table of contents for a quick explanation of each project and the relevant skills used.

---

## Table of Contents

| Project | Description | Skills |
|---|---|---|
| [Tidy Data Project](TidyData-Project/) | Tidying and analyzing U.S. federal R&D budget data (1976–2017) | Data wrangling, pandas, seaborn, matplotlib |
| [Streamlit App](basic_streamlit_app/) | Interactive data web application | Python, Streamlit |

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

---

*This repository was created by Eva S. Romero*
