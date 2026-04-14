# main_project.py

## Importing Libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Data splitting & preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz

## Evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, auc

## Setting up the page configuration 
st.set_page_config(page_title="Teaching Machines", page_icon="👾")
st.title("Teaching Machines: Interactive Supervised Learning")
st.markdown(
    "1) Upload your own dataset of interest. You can also choose from 4 sample datasets available:"
    "\n- [Iris](https://archive.ics.uci.edu/dataset/53/iris) — Classic flower measurements across 3 species"
    "\n- [Penguins](https://allisonhorst.github.io/palmerpenguins/articles/intro.html) — Palmer Station penguin measurements across 3 species"
    "\n- [Titanic](https://www.kaggle.com/competitions/titanic/data) — Passenger survival data from the 1912 disaster"
    "\n- [Tips](https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/tips.html) — Restaurant tipping behavior"
    "\n\n 2) Choose a target column — the variable you want the model to predict. Must be categorical (e.g. species, survived). Continuous columns like price or temperature won't work for classification."
    "\n\n 3) Choose which features to include. Features are the input variables the model uses to make predictions. You can include or exclude columns based on what you think is relevant."
    "\n\n 4) Select a train/test split. This evaluates the model's ability to generalize to new data rather than memorizing the training data. Many people use an 80/20 split."
    "\n\n 5) Choose a supervised ML model (Logistic Regression or Decision Tree) and tune its hyperparameters."
    "\n\n 6) Explore the model results and visualizations. Find anything interesting?"
)

df = None
y = None

## Setting up the sidebar
with st.sidebar:
    st.header("1. Pick your data")
    data_source = st.radio("Data Source", ["Sample Dataset", "Upload CSV File"])

    if data_source == "Sample Dataset":
        dataset_name = st.selectbox("Choose a dataset", ['titanic', 'penguins', 'iris', 'tips'])
        df = sns.load_dataset(dataset_name)
    else:
        upload = st.file_uploader("Upload a CSV file", type = ["csv"])
        if upload:
            df = pd.read_csv(upload)

    st.divider()

    st.header("2.  Choose a target")
    features = df.columns.tolist()
    y = st.selectbox("Select the target feature", features)
    df.dropna(subset=[y], inplace=True) # Handle missing values

    if pd.api.types.is_float_dtype(df[y]) and df[y].nunique() > 20:
        st.error(f"'{y}' looks like a continuous column. Please choose a categorical target for classification.")
        st.stop()

    if not pd.api.types.is_numeric_dtype(df[y]):
        le = LabelEncoder()
        df[y] = le.fit_transform(df[y].astype(str))

    st.header("3.  Choose the features to the predict model")
    X = st.multiselect("Select features", features, default=[f for f in features if f != y])

    # Preprocess X: drop rows with nulls in selected features, then encode categoricals
    df.dropna(subset=X, inplace=True)
    for col in X:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    st.divider()

    st.header("3.  Choose a Train/Test Split")
    test_size = st.slider("Set a test size (%) - Many use 20%", .10, .90, step = .10)
    st.markdown(f"{test_size*100}%")

    st.divider()

    st.header("4. Choose a model & tune the hyperparameters")
    model_name = st.selectbox("Choose a model", ["Logistic Regression", "Decision Tree"])

    if model_name == "Logistic Regression":
        st.info(
            "**Logistic Regression** works by drawing a straight boundary between classes and asking which " \
            "side of that line a sample falls on. Rather than just giving you a yes or no, it outputs a probability. "
            "You can see exactly how much each feature influenced the prediction. Works best when the relationship "
            "between features and the target is roughly linear."
        )
    else:
        st.info(
            "**Decision Trees** work by learning a series of yes/no questions about your data and following them down to a prediction, kind of like a flowchart you might draw by hand. One of its biggest strengths is that you can actually follow the logic and understand why it made a call, which is not something you can say about every model. It can also pick up on patterns that are not linear, which gives it an edge in more complex datasets. The tradeoff is that if you let it grow too deep, it starts memorizing the training data rather than learning from it. Use the sliders below to control complexity."
        )

    if model_name == "Decision Tree":
        max_depth = st.slider("Max depth", 1, 20, 5)
        min_split = st.slider("Min samples to split", 2, 20, 2)
        criterion = st.selectbox("Criterion", ["gini", "entropy"],
                                 help="Gini: measures how often a randomly chosen element would be misclassified. \n\nEntropy: measures the amount of disorder or impurity in a node.")
        
    st.divider()
    train_btn = st.button("Train Model", type="primary")

# Model training: 
if train_btn:
    if df is None or y is None:
        st.error("Please load a dataset and select a target feature.")
    else:
        # Split dataset into training and testing subsets
        X_train, X_test, y_train, y_test = train_test_split(df[X], df[y], test_size=test_size, random_state=42)
            
        if model_name == "Logistic Regression":
            # Initialize and train logistic regression model
            model = LogisticRegression()
            model.fit(X_train, y_train)
            # Predict on test data
            y_pred = model.predict(X_test)
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            # Extract coefficients and intercept
            coef = pd.Series(model.coef_[0], index=X)
            intercept = model.intercept_[0]
            # Get predicted probabilities for the test set
            probabilities = model.predict_proba(X_test)
            # Compute ROC AUC — binary uses a single probability column; multiclass uses OvR
            n_classes = len(model.classes_)
            if n_classes == 2:
                y_probs = probabilities[:, 1]
                fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_probs)
                roc_auc_lr = roc_auc_score(y_test, y_probs)
            else:
                y_probs = probabilities
                try:
                    roc_auc_lr = roc_auc_score(y_test, y_probs, multi_class='ovr')
                except ValueError:
                    roc_auc_lr = None
                fpr_lr, tpr_lr, thresholds_lr = None, None, None

            st.session_state["results"] = {
                "model_name": model_name,
                "accuracy": accuracy,
                "report": classification_report(y_test, y_pred, output_dict=True),
                "cm": cm,
                "classes": model.classes_,
                "y_test": y_test,
                "probabilities": probabilities,
                "y_probs": y_probs,
                "n_classes": n_classes,
                "fpr": fpr_lr,
                "tpr": tpr_lr,
                "roc_auc": roc_auc_lr,
                "coef": coef,
                "intercept": intercept,
                "feature_names": X,
                "feature_importances": None,
            }

        elif model_name == "Decision Tree":
            # Initialize and train tree classification model
            model = DecisionTreeClassifier(random_state=42, max_depth=max_depth, criterion=criterion, min_samples_split=min_split)
            model.fit(X_train, y_train)
            # Predict on test data
            y_pred = model.predict(X_test)
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            probabilities = model.predict_proba(X_test)
            # Compute ROC AUC — binary uses a single probability column; multiclass uses OvR
            n_classes = len(model.classes_)
            if n_classes == 2:
                y_probs = probabilities[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs)
            else:
                y_probs = probabilities
                try:
                    roc_auc = roc_auc_score(y_test, y_probs, multi_class='ovr')
                except ValueError:
                    roc_auc = None
                fpr, tpr, thresholds = None, None, None

            st.session_state["results"] = {
                "model_name": model_name,
                "accuracy": accuracy,
                "report": classification_report(y_test, y_pred, output_dict=True),
                "cm": cm,
                "classes": model.classes_,
                "y_test": y_test,
                "probabilities": probabilities,
                "y_probs": y_probs,
                "n_classes": n_classes,
                "fpr": fpr,
                "tpr": tpr,
                "roc_auc": roc_auc,
                "coef": None,
                "intercept": None,
                "feature_names": X,
                "feature_importances": model.feature_importances_,
                "model": model,
            }

results = st.session_state.get("results")

## Tabs
tab_data, tab_results, tab_viz = st.tabs(["Data Preview", "Model Results", "Visualizations"])

# Tab 1: Data Preview
with tab_data:
    if df is None:
        st.info("Load a dataset from the sidebar to get started.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values", int(df.isnull().sum().sum()))
        st.dataframe(df.head(10))

        with st.expander("Column types & missing values"):
            st.dataframe(
                pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isnull().sum()}),
                use_container_width=False
            )

        if y:
            st.subheader(f"Target distribution: `{y}`")
            fig, ax = plt.subplots(figsize=(5, 3))
            df[y].value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
            ax.set(xlabel="Class", ylabel="Count")
            st.pyplot(fig)
            plt.close(fig)


# Tab 2: Model Results
with tab_results:
    if results is None:
        st.info("Train a model to see results here.")
    else:
        st.subheader(f"Results — {results['model_name']}")
        c1, c2 = st.columns(2)
        c1.metric("Test Accuracy", f"{results['accuracy']:.3f}")
        c1.caption("The proportion of data points the model predicted correctly. Closer to 1.0 is better, but accuracy alone can be misleading when classes are imbalanced.")
        if results["roc_auc"] is not None:
            c2.metric("ROC AUC Score", f"{results['roc_auc']:.3f}")
            c2.caption("Area Under the ROC Curve. Measures how well the model separates classes regardless of threshold. 0.5 = random guessing, 1.0 = perfect separation.")

        st.subheader("Classification Report")
        st.dataframe(
            pd.DataFrame(results["report"]).transpose().style.format(precision=3),
            use_container_width=False
        )
        with st.expander("How to read the classification report"):
            st.markdown(
                "- **Precision** — Of all the samples the model predicted as a given class, how many actually belonged to that class?\n"
                "- **Recall** — Of all the samples that truly belonged to a given class, how many did the model actually catch? High recall means the model is not missing much.\n"
                "- **F1-score** —  Precision and recall often trade off against each other, so the F1-score combines them into one number by taking their harmonic mean. It is especially useful when your classes are imbalanced, because it forces the model to do well on both metrics, not just one.\n"
                "- **Support** — The number of real samples from each class in your test set. This gives you important context. A 90% recall score means something very different if it is based on 5 samples versus 500.\n"
                "- **Macro avg** — The average of each metric calculated across all classes, where every class is weighted equally regardless of how many samples it has. This is useful when you care about performance on every class, even rare ones.\n"
                "- **Weighted avg** — The same averages, but each class is weighted by how many samples it contributed to the test set. This tends to reflect overall real-world performance better when your classes are not evenly distributed."
            )

        if results["coef"] is not None:
            st.subheader("Model Coefficients")
            coef_df = pd.DataFrame({
                "Feature": results["feature_names"],
                "Coefficient": results["coef"].values
            }).sort_values("Coefficient", key=abs, ascending=False)
            st.dataframe(coef_df, use_container_width=False)
            st.caption("Each coefficient tells you how much a one unit increase in that feature moves the model toward or away from predicting the positive class. A large positive number means that feature is pushing the model to say yes. A large negative number means it is pushing the model to say no. Features with coefficients close to zero are not doing much either way.")
            st.metric("Intercept", f"{results['intercept']:.4f}")
            st.caption("The baseline prediction when all feature values are zero.")

        if results["feature_importances"] is not None:
            st.subheader("Feature Importances")
            importance_df = pd.DataFrame({
                "Feature": results["feature_names"],
                "Importance": results["feature_importances"],
            }).sort_values("Importance", ascending=False)
            fig, ax = plt.subplots(figsize=(7, max(3, len(importance_df) * 0.4)))
            sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax, color="steelblue")
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Shows how much each feature contributed to the model's decisions. Higher importance means the feature was used more often and more effectively to split the data into correct groups.")

# Tab 3: Visualizations
with tab_viz:
    if results is None:
        st.info("Train a model to see visualizations here.")
    else:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        n_cm = results["cm"].shape[0]
        ConfusionMatrixDisplay(results["cm"], display_labels=results["classes"][:n_cm]).plot(ax=ax, colorbar=False)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("A confusion matrix shows you exactly where your model is getting things right and where it is getting things wrong. Each row represents the actual class of a sample, and each column represents what the model predicted. The numbers along the diagonal are correct predictions. Everything off the diagonal is a mistake. This is useful because overall accuracy can be misleading. A model can look 90% accurate on paper but be completely failing on one class if that class is rare. The confusion matrix breaks that down so you can see the full picture.")

        st.subheader("Predicted Probabilities")
        prob_df = pd.DataFrame(
            results["probabilities"],
            columns=[f"Class {c}" for c in results["classes"]]
        )
        st.dataframe(prob_df.style.format(precision=4), use_container_width=False)
        st.caption("Think of these as the model's 'confidence scores.' For each test sample, the model assigns a score to every possible class, and those scores always sum to 1. It predicts whichever class scored highest. Scores close to 0.5 on both sides mean the model is on the fence. Scores close to 1.0 mean it is highly confident. Checking these probabilities rather than just the final prediction gives you a much fuller picture of how your model is actually performing.")
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(6, 5))
        y_test_arr = results["y_test"].to_numpy()
        if results["n_classes"] == 2:
            fpr, tpr = results["fpr"], results["tpr"]
            ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
        else:
            y_bin = label_binarize(y_test_arr, classes=results["classes"])
            for i, cls in enumerate(results["classes"]):
                fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], results["probabilities"][:, i])
                ax.plot(fpr_i, tpr_i, label=f"Class {cls} (AUC = {auc(fpr_i, tpr_i):.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("The ROC curve is a visual representation of model performance across all thresholds. It is drawn by catching the tradeoff between the true positive rate (TPR) and false positive rate (FPR) across all possible thresholds. "
                   "The AUC (area under the curve) represents the models ability to distinguish between classes and its probability of ranking a random positive example higher than a random negative one. Generally, the model with greater AUC is ideal when comparing model performance, closer to 1.0 is better")

        if results["model_name"] == "Decision Tree" and results.get("model") is not None:
            st.subheader("Decision Tree Structure")
            dot_data = export_graphviz(
                results["model"],
                feature_names=results["feature_names"],
                class_names=[str(c) for c in results["classes"]],
                filled=True,
                rounded=True,
                special_characters=True,
            )
            st.graphviz_chart(dot_data)
            st.caption("Each node shows the feature and threshold used to split the data. The color indicates the dominant class at that node — deeper color means higher confidence. Follow the branches from top to bottom to trace how the model reaches a prediction.")
