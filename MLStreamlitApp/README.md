# Teaching Machines: Interactive Supervised Learning App

## Overview
This project is an interactive Streamlit web app that lets users explore supervised machine learning
without writing any code. Users can load a dataset, choose a target variable, select features, tune
model hyperparameters, and immediately see results and visualizations, all through a point-and-click interface.

**Supervised Learning:** A type of machine learning where the model learns from labeled examples —
data where the correct answer is already known, and uses those patterns to make predictions on new data.
This app focuses on **classification**, where the goal is to predict which category a sample belongs to.

## How to Run
1. Clone the repository
2. Install dependencies (see Requirements below)
3. Run the app with:
```bash
streamlit run MLStreamlitApp/main_project.py
```
4. The app will open in your browser automatically

## Requirements
Install dependencies with:
```bash
pip install streamlit pandas seaborn matplotlib scikit-learn graphviz
```

| Library | Purpose |
|---|---|
| `streamlit` | Web app framework and interactive UI |
| `pandas` | Data loading and manipulation |
| `seaborn` | Loading sample datasets and visualizations |
| `matplotlib` | Plot rendering |
| `scikit-learn` | ML models, preprocessing, and evaluation metrics |
| `graphviz` | Decision tree structure visualization |

## App Features

### Data
- Load one of 4 built-in sample datasets or upload your own CSV
- Built-in datasets: [Iris](https://archive.ics.uci.edu/dataset/53/iris), [Penguins](https://allisonhorst.github.io/palmerpenguins/articles/intro.html), [Titanic](https://www.kaggle.com/competitions/titanic/data), [Tips](https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/tips.html)
- Data preview tab shows row/column counts, missing values, and target class distribution

### Models
| Model | Description |
|---|---|
| Logistic Regression | Draws a linear boundary between classes; outputs class probabilities |
| Decision Tree | Learns yes/no rules to classify data; supports depth and split tuning |

### Outputs
- **Model Results tab:** accuracy, ROC AUC, classification report, model coefficients (LR) or feature importances (DT)
- **Visualizations tab:** confusion matrix, predicted probabilities, ROC curve, decision tree diagram (DT only)
- Inline explanations throughout to help interpret every metric and chart

## Pre-processing Steps
1. Rows with missing values in the target or selected features are dropped
2. Categorical columns are encoded using `LabelEncoder`
3. Continuous columns selected as the target trigger an error with a clear message
4. Train/test split is configurable from 10% to 90%

## References
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Palmer Penguins Dataset](https://allisonhorst.github.io/palmerpenguins/)
- [Titanic Dataset — Kaggle](https://www.kaggle.com/competitions/titanic/data)
- [Iris Dataset — UCI ML Repository](https://archive.ics.uci.edu/dataset/53/iris)
