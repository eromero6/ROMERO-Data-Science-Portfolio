# final_project.py

# Importing libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data splitting and preprocessing
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, accuracy_score

# Hierarchical clustering utilities
from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram
import plotly.express as px

# Dataset
from sklearn.datasets import load_breast_cancer


## Setting up page configuration
st.set_page_config(page_title="Learning without Supervision", page_icon="📊")

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

def scroll_to_top():
    st.components.v1.html("""
        <script>
            function doScroll() {
                var el = window.parent.document.querySelector('section[data-testid="stMain"]')
                      || window.parent.document.querySelector('section.main')
                      || window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                if (el) el.scrollTop = 0;
            }
            setTimeout(doScroll, 50);
        </script>
    """, height=0)

@st.cache_data
def load_democracy_data():
    url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2024/2024-11-05/democracy_data.csv"
    df_full = pd.read_csv(url)
    data = df_full[df_full["year"] == 2020].copy()
    bool_cols = ["is_monarchy", "is_democracy", "is_presidential", "is_colony", "is_communist",
                 "has_proportional_voting", "has_full_suffrage", "is_multiparty",
                 "has_free_and_fair_election", "has_alternation", "has_new_constitution"]
    num_cols = ["regime_category_index", "spatial_democracy", "parliament_chambers",
                "lower_house_members", "spatial_electoral", "electoral_category_index"]
    for col in bool_cols:
        data[col] = data[col].map({True: 1, False: 0})
    feature_cols = num_cols + bool_cols
    features_df = data[feature_cols].copy()
    mask = features_df.notna().all(axis=1)
    features_df = features_df[mask].reset_index(drop=True)
    country_names = data.loc[mask, "country_name"].values
    country_codes = data.loc[mask, "country_code"].values
    return features_df, country_names, country_codes

# Home page
def show_home():
    st.title("Learning Without Supervision 📊")
    st.markdown("This is an interactive unsupervised machine learning app. It is designed to teach users about the following ML models: Principal Component Analysis (PCA), K-Means Clustering, and Hierarchical Clustering.")

    st.markdown("### What is Unsupervised Machine Learning?")
    st.markdown("""
                    Unsupervised machine learning is a type of algorithm that finds patterns in data
                    **without any predefined labels or correct answers** to learn from. Unlike supervised
                    learning — where a model is trained on labeled examples — unsupervised methods are
                    given raw data and tasked with discovering its underlying structure on their own.
                
                **Common tasks in unsupervised learning include:**
                - **Clustering** — grouping similar observations together (e.g., customer segmentation)
                - **Dimensionality reduction** — compressing many variables into fewer while preserving structure (e.g., PCA)
                - **Anomaly detection** — identifying observations that don't fit the general pattern
                
                **Why it matters:** Real-world data is often unlabeled, making unsupervised learning a
                powerful tool for exploration — surfacing patterns, relationships, and structure that
                might not be obvious from the raw data alone.
                """)

    st.markdown("### Instructions")
    st.markdown("Pick from one of three models available to explore. From there, you will be prompted to upload your own dataset, or use the example data provided.")

    st.markdown("---")
    st.markdown("## Choose a Model")

    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] { align-items: stretch; }
    div[data-testid="stColumn"] > div:first-child { height: 100%; }
    div[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"] { height: 100%; }
    div[data-testid="stColumn"] [data-testid="stVerticalBlock"] {
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    </style>
    """, unsafe_allow_html=True) ## Needed help aligning the widgets because they were not lining up at all

    # Creating 3 columns to choose ML model
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔵 Principal Component Analysis")
        st.markdown("Reduce the dimensionality of your data while preserving the most important variance.")
        if st.button("Explore PCA", use_container_width=True):
            st.session_state.page = "pca"
            st.rerun()

    with col2:
        st.markdown("### 🟢 K-Means Clustering")
        st.markdown("Partition your data into K distinct clusters based on feature similarity")
        if st.button("Explore K-Means", use_container_width=True):
            st.session_state.page = "kmeans"
            st.rerun()

    with col3:
        st.markdown("### 🟠 Hierarchical Clustering")
        st.markdown("Build a tree of clusters to reveal nested groupings in your data.")
        if st.button("Explore Hierarchical Clustering", use_container_width=True):
            st.session_state.page = "hierarchical"
            st.rerun()

# PCA Page
def show_pca():
    scroll_to_top()
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("Principal Component Analysis 🔵")
    st.markdown("### What is PCA?")
    st.markdown("PCA is a dimensionality-reduction method, used to simplify complex datasets while minimizing information loss."
                " It transforms correlated variables into a smaller set called principal components, which capture the total amount of variance in the data."
                " They are ordered by the amount of variance explained; the first component capturing the most variance, with subsequent ones capturing less."
                " This helps improve overall computational efficiency, enhances data visualization, and prepares data for better model performance.")
    st.markdown("### 1) Choose a dataset")
    data_source = st.radio("Data Source", ["Sample Dataset", "Upload CSV File"])

    # SAMPLE DATASET CODE
    if data_source == "Sample Dataset":
        st.markdown("You have selected the Breast Cancer Wisconsin dataset!")
        st.markdown("""
        The **Breast Cancer Wisconsin dataset** contains measurements from 569 digitized fine needle aspirate (FNA)
        biopsy samples. Each row is a tumor described by **30 numeric features** — such as radius, texture, perimeter,
        area, and smoothness — computed from cell nuclei in the image. The target label classifies each tumor as
        **malignant (212 cases)** or **benign (357 cases)**.

        It is a classic benchmark dataset for unsupervised learning because the two groups are naturally separable
        in the feature space — making it ideal for demonstrating how well algorithms like PCA and K-Means can
        recover real structure without ever seeing the labels.
        """)
        breast_cancer = load_breast_cancer()

        # Raw data
        st.markdown("### 2) Raw Data Preview")
        X = breast_cancer.data
        y = breast_cancer.target
        feature_names = breast_cancer.feature_names
        target_names = breast_cancer.target_names
        df_preview = pd.DataFrame(X, columns=feature_names)
        st.dataframe(df_preview.head())
        st.divider()

        # Standardized data
        st.markdown("### 3) Standardize the Data")
        st.markdown("PCA is sensitive to the scale of the variables, therefore we must center and scale the data. "
                    "Features are centered (mean = 0) and scaled (std = 1) so no single feature dominates the principal components.")
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        df_std = pd.DataFrame(X_std, columns=feature_names)
        st.dataframe(df_std.head())
        st.divider()
 
        # PCA results
        st.markdown("### 4) PCA Results")
        st.markdown("In this step, we compute the PCA on the standardized data. PCA essentially 'rotates' the data"
                    " to align new axes that maximize the variance of the data, and then compresses it to keep the most important information.")
        
        st.markdown("#### How it works?")
        st.markdown("The first principal component faces the direction along which the data varies most. The principal components that follow are uncorrelated"
                    " to the previous ones and capture the remaining variance.")
        st. markdown("For this demonstration, the data is reduced to 2 dimensions, allowing us to create a 2D plot that illustrates the"
                     " distribution of data points.")
        st.divider()
        
        # Results metrics
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_std)
        explained_variance = pca.explained_variance_ratio_

        col_a, col_b = st.columns(2)
        col_a.metric("PC1 Variance Explained", f"{explained_variance[0]*100:.1f}%")
        col_b.metric("PC2 Variance Explained", f"{explained_variance[1]*100:.1f}%")
        st.markdown(f"**Combined:** {sum(explained_variance)*100:.1f}% of total variance captured by 2 components.")

        viz_choice = st.radio("Choose a visualization", ["Scatter Plot", "PCA Loadings", "Scree Plot"], horizontal=True)

        # Scatter plot
        if viz_choice == "Scatter Plot":
            st.markdown("Where do the observations land in the new coordinate system defined by the first 2 principal components? Can you see distinct groups emerge?")
            fig, ax = plt.subplots()
            for label, name in enumerate(target_names):
                mask = y == label
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, edgecolors='k', alpha=0.6, s=40)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title("PCA — 2D Projection of Breast Cancer Dataset")
            ax.legend()
            st.pyplot(fig)

        # PCA loadings
        elif viz_choice == "PCA Loadings":
            st.markdown(" This chart shows how much each original variable contributes to the first two principal components (PC1 and PC2) extracted from the dataset.")
            loadings_df = pd.DataFrame(
                pca.components_,
                columns=feature_names,
                index=[f'PC{i+1}' for i in range(pca.n_components_)]
            )
            features = loadings_df.columns.tolist()
            y_pos = np.arange(len(features))
            bar_height = 0.3

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.barh(y_pos + bar_height/2, loadings_df.loc['PC1'], bar_height,
                    label='PC1', color="#ed34c2", edgecolor='none')
            ax.barh(y_pos - bar_height/2, loadings_df.loc['PC2'], bar_height,
                    label='PC2', color="#075779", edgecolor='none')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Loading Weight')
            ax.set_title('PCA Loadings (how each stat contributes)', fontweight='bold', loc='left')
            ax.axvline(0, color='grey', linewidth=0.8)
            ax.legend(loc='upper right', frameon=True)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            ax.set_frame_on(False)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("#### How to Read This PCA Loadings Chart")
            st.markdown("""
                            **The bars** represent loading weights — the correlation between each original variable
                            and a given principal component. Each variable has two bars: one for PC1 (pink)
                            and one for PC2 (dark navy).

                            **Bar direction** tells you the sign of the relationship. Bars extending to the right
                            (positive loadings) mean the variable moves in the same direction as the component;
                            bars extending to the left (negative loadings) mean the opposite direction.

                            **Bar length** reflects the strength of contribution. Longer bars indicate a variable
                            has more influence on that component, while shorter bars near zero contribute very little.

                            **PC1 vs. PC2** capture different sources of variation in the data. Variables with long
                            PC1 bars drive the primary axis of variation in the dataset. Variables with long PC2
                            bars — but short PC1 bars — describe a secondary, independent pattern that PC1 doesn't capture.

                            **What to look for:** Variables that load heavily on the same component and in the same
                            direction tend to move together and represent a shared underlying signal. Variables with
                            strong loadings on PC2 but weak ones on PC1 are capturing something dimensionally distinct
                            from the main trend.
                            """)

        # Scree Plot
        elif viz_choice == "Scree Plot":
            st.markdown("A scree plot shows the explained variance for each principal component. Look for the 'elbow' where adding more components gives diminishing returns.")

            n_components = st.slider("Number of Principal Components", min_value=1, max_value=15, value=2)

            pca_full = PCA(n_components=15).fit(X_std)
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

            label = "component" if n_components == 1 else "components"
            st.metric(f"Variance Explained by {n_components} {label}", f"{cumulative_variance[n_components - 1]*100:.1f}%")

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(range(1, 16), cumulative_variance, marker='o')
            ax.axvline(n_components, color='red', linestyle='--', alpha=0.6)
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Explained Variance')
            ax.set_title('PCA Variance Explained')
            ax.set_xticks(range(1, 16))
            ax.grid(True)
            st.pyplot(fig)

            st.markdown("#### How to Read This Scree Plot")
            st.markdown("""
                            This chart displays the **cumulative proportion of variance** explained as more principal
                            components are added. Each point on the curve represents the total variance captured by
                            that many components combined — not just the component itself.

                            **The y-axis** ranges from 0 to 1 (or 0% to 100%) and represents how much of the
                            dataset's total information is retained. A value of 0.90, for example, means 90% of
                            the variation in the original data is preserved.

                            **The x-axis** shows the number of principal components included, starting from 1
                            and increasing sequentially.

                            **The curve shape** is always non-decreasing — each additional component can only
                            add explained variance, never subtract it. The curve typically rises steeply at first,
                            then flattens as later components contribute diminishing returns.

                            **The elbow** is the key feature to identify: the point where the curve transitions
                            from steep to gradual. Components before the elbow each add meaningful variance;
                            components after it offer only marginal gains. The number of components at the elbow
                            is a common, practical choice for how many to retain.

                            **What to look for:** A lower number of components that together capture a high
                            proportion of variance (commonly 80–95%) is a sign that the dataset's structure can
                            be well-represented in a reduced dimensional space — making downstream analysis
                            simpler without significant loss of information.
                            """)

        st.markdown("---")
        st.markdown("### Key Takeaways")
        pc1_pct = explained_variance[0] * 100
        pc2_pct = explained_variance[1] * 100
        combined_pct = pc1_pct + pc2_pct
        st.info(f"""
**2D scatter plot reliability:** PC1 ({pc1_pct:.1f}%) + PC2 ({pc2_pct:.1f}%) = **{combined_pct:.1f}% of total variance** captured in the 2D projection.
{"✅ The scatter plot faithfully represents the data's structure — most variation is visible in 2D." if combined_pct >= 60 else "⚠️ Less than 60% of variance is visible in 2D — the scatter plot gives a partial picture. Use the Scree Plot to determine how many components are needed for a fuller representation."}

**Dominant component:** PC1 alone explains {pc1_pct:.1f}% of total variance, meaning a single axis captures the largest source of variation in the 30 features of this dataset.

**PCA Loadings:** Check which features have the longest bars in PC1 and PC2 — these are the variables driving the separation you see in the scatter plot.

**Recommendation:** {"The dataset compresses well into 2D. PCA is a strong fit here." if combined_pct >= 60 else "Retain more components (see Scree Plot) for downstream analysis, as the 2D view alone misses meaningful structure."}
""")
        st.divider()

    # Uploading CSV dataset
    else:
        st.markdown("""
        **What kind of dataset should you upload?**
        - Feature columns must be **numeric** (integers or decimals).
        - If you have a **label or target column** (e.g. a class or category), keep it — you can select it below to color the scatter plot.
        - Remove any **ID or index columns** — these are not measurements and will distort the results.
        - Your file must have **at least 2 numeric feature columns** for PCA to work.
        - Rows with missing values in feature columns will be dropped automatically.
        """)

        upload = st.file_uploader("Upload a CSV file", type=["csv"])

        if upload:
            df_raw = pd.read_csv(upload)

            label_col = st.selectbox(
                "Optional: select a column to color the scatter plot by (e.g. a class or target column):",
                options=["None"] + df_raw.columns.tolist(),
                key="upload_label_col"
            )

            df_features = df_raw.drop(columns=[label_col]) if label_col != "None" else df_raw.copy()
            df_numeric = df_features.select_dtypes(include=[np.number])
            valid_mask = df_numeric.notna().all(axis=1)
            df_numeric = df_numeric[valid_mask].reset_index(drop=True)
            color_series = df_raw[label_col][valid_mask].reset_index(drop=True) if label_col != "None" else None

            if df_numeric.shape[1] < 2:
                st.error("Your dataset must contain at least 2 numeric feature columns for PCA. Please check your file and re-upload.")
            else:
                feature_names_up = df_numeric.columns.tolist()
                X_up = df_numeric.values

                st.markdown("### 2) Raw Data Preview")
                st.dataframe(df_numeric.head())
                st.divider()

        # Standardize data
                st.markdown("### 3) Standardize the Data")
                st.markdown("PCA is sensitive to the scale of the variables, therefore we must center and scale the data. "
                            "Features are centered (mean = 0) and scaled (std = 1) so no single feature dominates the principal components.")
                scaler_up = StandardScaler()
                X_std_up = scaler_up.fit_transform(X_up)
                df_std_up = pd.DataFrame(X_std_up, columns=feature_names_up)
                st.dataframe(df_std_up.head())
                st.divider()

        # PCA results
                st.markdown("### 4) PCA Results")
                st.markdown("In this step, we compute the PCA on the standardized data. PCA essentially 'rotates' the data"
                            " to align new axes that maximize the variance of the data, and then compresses it to keep the most important information.")
               
                st.markdown("#### How it works?")
                st.markdown("The first principal component faces the direction along which the data varies most. The principal components that follow are uncorrelated"
                            " to the previous ones and capture the remaining variance.")
                st. markdown("For this demonstration, the data is reduced to 2 dimensions, allowing us to create a 2D plot that illustrates the"
                             " distribution of data points.")
                st.divider()

        # Result metrics
                pca_up = PCA(n_components=2)
                X_pca_up = pca_up.fit_transform(X_std_up)
                ev_up = pca_up.explained_variance_ratio_

                col_a, col_b = st.columns(2)
                col_a.metric("PC1 Variance Explained", f"{ev_up[0]*100:.1f}%")
                col_b.metric("PC2 Variance Explained", f"{ev_up[1]*100:.1f}%")
                st.markdown(f"**Combined:** {sum(ev_up)*100:.1f}% of total variance captured by 2 components.")

                viz_choice_up = st.radio("Choose a visualization", ["Scatter Plot", "PCA Loadings", "Scree Plot"],
                                         horizontal=True, key="upload_viz")

        # Scatter plot
                if viz_choice_up == "Scatter Plot":
                    st.markdown("Where do the observations land in the new coordinate system defined by the first 2 principal components? Can you see distinct groups emerge?")
                    if color_series is not None:
                        if color_series.nunique() <= 20:
                            st.info(f"**Colored by group:** '{label_col}' has {color_series.nunique()} unique values, so each group gets its own color and a legend entry.")
                        else:
                            st.info(f"**Colored by gradient:** '{label_col}' is a continuous variable with {color_series.nunique()} unique values, so a color gradient (colorbar) is used instead of a legend.")
                    fig, ax = plt.subplots()
                    if color_series is not None:
                        if color_series.nunique() <= 20:
                            # Categorical: one color per group with a legend
                            for lbl in color_series.unique():
                                mask = color_series == lbl
                                ax.scatter(X_pca_up[mask, 0], X_pca_up[mask, 1], label=str(lbl),
                                           alpha=0.6, s=40, edgecolors='k')
                            ax.legend(title=label_col)
                        else:
                            # Continuous: use a colormap with a colorbar
                            sc = ax.scatter(X_pca_up[:, 0], X_pca_up[:, 1],
                                            c=color_series, cmap='viridis', alpha=0.6, s=40)
                            plt.colorbar(sc, ax=ax, label=label_col)
                    else:
                        ax.scatter(X_pca_up[:, 0], X_pca_up[:, 1], alpha=0.6, s=40, edgecolors='k')
                    ax.set_xlabel("Principal Component 1")
                    ax.set_ylabel("Principal Component 2")
                    ax.set_title("PCA — 2D Projection of Uploaded Dataset")
                    st.pyplot(fig)

        # PCA loadings
                elif viz_choice_up == "PCA Loadings":
                    st.markdown("This chart shows how much each original variable contributes to the first two principal components (PC1 and PC2).")
                    loadings_df_up = pd.DataFrame(
                        pca_up.components_,
                        columns=feature_names_up,
                        index=[f'PC{i+1}' for i in range(pca_up.n_components_)]
                    )
                    features_up = loadings_df_up.columns.tolist()
                    y_pos_up = np.arange(len(features_up))
                    bar_height = 0.3

                    fig, ax = plt.subplots(figsize=(10, max(5, len(features_up) * 0.4)))
                    ax.barh(y_pos_up + bar_height/2, loadings_df_up.loc['PC1'], bar_height,
                            label='PC1', color="#ed34c2", edgecolor='none')
                    ax.barh(y_pos_up - bar_height/2, loadings_df_up.loc['PC2'], bar_height,
                            label='PC2', color="#075779", edgecolor='none')
                    ax.set_yticks(y_pos_up)
                    ax.set_yticklabels(features_up)
                    ax.set_xlabel('Loading Weight')
                    ax.set_title('PCA Loadings (how each feature contributes)', fontweight='bold', loc='left')
                    ax.axvline(0, color='grey', linewidth=0.8)
                    ax.legend(loc='upper right', frameon=True)
                    ax.invert_yaxis()
                    ax.grid(axis='x', alpha=0.3)
                    ax.set_frame_on(False)
                    plt.tight_layout()
                    st.pyplot(fig)

                    st.markdown("#### How to Read This PCA Loadings Chart")
                    st.markdown("""
                            **The bars** represent loading weights — the correlation between each original variable
                            and a given principal component. Each variable has two bars: one for PC1 (pink)
                            and one for PC2 (dark navy).

                            **Bar direction** tells you the sign of the relationship. Bars extending to the right
                            (positive loadings) mean the variable moves in the same direction as the component;
                            bars extending to the left (negative loadings) mean the opposite direction.

                            **Bar length** reflects the strength of contribution. Longer bars indicate a variable
                            has more influence on that component, while shorter bars near zero contribute very little.

                            **PC1 vs. PC2** capture different sources of variation in the data. Variables with long
                            PC1 bars drive the primary axis of variation in the dataset. Variables with long PC2
                            bars — but short PC1 bars — describe a secondary, independent pattern that PC1 doesn't capture.

                            **What to look for:** Variables that load heavily on the same component and in the same
                            direction tend to move together and represent a shared underlying signal. Variables with
                            strong loadings on PC2 but weak ones on PC1 are capturing something dimensionally distinct
                            from the main trend.
                            """)

        # Scree Plot
                elif viz_choice_up == "Scree Plot":
                    st.markdown("A scree plot shows the explained variance for each principal component. Look for the 'elbow' where adding more components gives diminishing returns.")
                    max_components = min(15, X_std_up.shape[1], X_std_up.shape[0])
                    n_components_up = st.slider("Number of Principal Components", min_value=1,
                                                max_value=max_components, value= 2, key="upload_scree_slider")
                    pca_full_up = PCA(n_components=max_components).fit(X_std_up)
                    cumvar_up = np.cumsum(pca_full_up.explained_variance_ratio_)

                    label_up = "component" if n_components_up == 1 else "components"
                    st.metric(f"Variance Explained by {n_components_up} {label_up}", f"{cumvar_up[n_components_up - 1]*100:.1f}%")

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(range(1, max_components + 1), cumvar_up, marker='o')
                    ax.axvline(n_components_up, color='red', linestyle='--', alpha=0.6)
                    ax.set_xlabel('Number of Components')
                    ax.set_ylabel('Cumulative Explained Variance')
                    ax.set_title('PCA Variance Explained')
                    ax.set_xticks(range(1, max_components + 1))
                    ax.grid(True)
                    st.pyplot(fig)

                    st.markdown("#### How to Read This Scree Plot")
                    st.markdown("""
                            This chart displays the **cumulative proportion of variance** explained as more principal
                            components are added. Each point on the curve represents the total variance captured by
                            that many components combined — not just the component itself.

                            **The y-axis** ranges from 0 to 1 (or 0% to 100%) and represents how much of the
                            dataset's total information is retained. A value of 0.90, for example, means 90% of
                            the variation in the original data is preserved.

                            **The x-axis** shows the number of principal components included, starting from 1
                            and increasing sequentially.

                            **The curve shape** is always non-decreasing — each additional component can only
                            add explained variance, never subtract it. The curve typically rises steeply at first,
                            then flattens as later components contribute diminishing returns.

                            **The elbow** is the key feature to identify: the point where the curve transitions
                            from steep to gradual. Components before the elbow each add meaningful variance;
                            components after it offer only marginal gains. The number of components at the elbow
                            is a common, practical choice for how many to retain.

                            **What to look for:** A lower number of components that together capture a high
                            proportion of variance (commonly 80–95%) is a sign that the dataset's structure can
                            be well-represented in a reduced dimensional space — making downstream analysis
                            simpler without significant loss of information.
                            """)

                st.markdown("---")
                st.markdown("### Key Takeaways")
                pc1_pct_up = ev_up[0] * 100
                pc2_pct_up = ev_up[1] * 100
                combined_pct_up = pc1_pct_up + pc2_pct_up
                st.info(f"""
**2D scatter plot reliability:** PC1 ({pc1_pct_up:.1f}%) + PC2 ({pc2_pct_up:.1f}%) = **{combined_pct_up:.1f}% of total variance** captured in the 2D projection.
{"✅ The scatter plot faithfully represents your data's structure — most variation is visible in 2D." if combined_pct_up >= 60 else "⚠️ Less than 60% of variance is captured in 2D. The scatter plot is a partial view — use the Scree Plot to choose a better number of components for downstream analysis."}

**Dominant component:** PC1 alone explains {pc1_pct_up:.1f}% of total variance — it represents the single largest source of variation across all your features.

**PCA Loadings:** Review the loading chart to identify which of your features contribute most to PC1 and PC2. These are the variables driving any separation or grouping visible in the scatter plot.

**Recommendation:** {"Your data compresses well into 2 dimensions. If you plan to use PCA for downstream modeling, 2 components may be sufficient." if combined_pct_up >= 60 else f"Consider retaining more components (see Scree Plot) to preserve at least 80% of variance before using PCA for downstream modeling."}
""")
                st.divider()

    if st.button("← Back to Home", key="pca_back_bottom"):
        st.session_state.page = "home"
        st.rerun()


def show_kmeans():
    scroll_to_top()
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("K-Means Clustering 🟢")
    st.markdown("### What is K-Means Clustering?")
    st.markdown("""
                    K-Means works by **partitioning data into a fixed number of groups (k)**, where each observation is assigned
                    to the cluster whose center it is closest to.

                    The algorithm follows a simple iterative process:
                    1. **Initialize** — randomly place k centroids (cluster centers) in the data.
                    2. **Assign** — assign each data point to its nearest centroid.
                    3. **Update** — recalculate each centroid as the mean of all points assigned to it.
                    4. **Repeat** — continue until the cluster assignments stop changing significantly, or maximum iterations are reached.

                    **The result** is k distinct clusters, where points within the same group are more
                    similar to each other than to points in other groups.

                    **Choosing k** is one of the key decisions in K-Means. A common approach is the
                    **elbow method** — running the algorithm for different values of k and looking for
                    the point where adding more clusters stops meaningfully reducing the within-cluster variance.

                    **Keep in mind:** K-Means assumes clusters are roughly spherical and similar in size,
                    so it works best when those conditions are reasonably met in the data.
                    """)

    st.markdown("### 1) Choose a dataset")
    data_source = st.radio("Data Source", ["Sample Dataset", "Upload CSV File"])

    # SAMPLE DATASET CODE
    if data_source == "Sample Dataset":
        st.markdown("You have selected the Breast Cancer Wisconsin dataset!")
        st.markdown("""
        The **Breast Cancer Wisconsin dataset** contains measurements from 569 digitized fine needle aspirate (FNA)
        biopsy samples. Each row is a tumor described by **30 numeric features** — such as radius, texture, perimeter,
        area, and smoothness — computed from cell nuclei in the image. The target label classifies each tumor as
        **malignant (212 cases)** or **benign (357 cases)**.

        It is a classic benchmark dataset for unsupervised learning because the two groups are naturally separable
        in the feature space — making it ideal for demonstrating how well algorithms like PCA and K-Means can
        recover real structure without ever seeing the labels.
        """)
        breast_cancer = load_breast_cancer()

        # Raw data
        st.markdown("### 2) Raw Data Preview")
        X = breast_cancer.data
        y = breast_cancer.target
        feature_names = breast_cancer.feature_names
        target_names = breast_cancer.target_names
        df_preview = pd.DataFrame(X, columns=feature_names)
        st.dataframe(df_preview.head())
        st.divider()

        # Standardized data
        st.markdown("### 3) Standardize the Data")
        st.markdown(" K-Means relies on distance calculations and can be biased by the scale of features. "
                    "Therefore, we must center (mean = 0) and scale (std = 1) the data.")
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        df_std = pd.DataFrame(X_std, columns=feature_names)
        st.dataframe(df_std.head())
        st.divider()

        # Computing KMeans clustering
        st.markdown("### 4) K-Means Results")
        st.markdown("Adjust the slider to choose k — the number of clusters — and see how the results change.")

        k = st.slider("Number of Clusters (k)", min_value=2, max_value=10, value=2, key="kmeans_k")

        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_std)

        pca_km = PCA(n_components=2)
        X_pca_km = pca_km.fit_transform(X_std)

        sil = silhouette_score(X_std, clusters)
        acc = accuracy_score(y, clusters)
        acc = max(acc, 1 - acc)  # correct for label inversion

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Clusters (k)", k)
        col_b.metric("Inertia", f"{kmeans.inertia_:,.1f}")
        col_c.metric("Silhouette Score", f"{sil:.3f}")

        st.markdown("""
        **Understanding the metrics:**
        - **Clusters (k):** The number of groups you asked K-Means to find. There is no single correct value —
          the goal is to choose a k that produces meaningful, well-separated groups in your data.
        - **Inertia:** The total sum of squared distances between each point and its cluster center.
          Lower inertia means points sit closer to their cluster centers. It will always decrease as k increases,
          so use it alongside the Silhouette Score rather than in isolation. 
        - **Silhouette Score:** Measures how well each point fits its own cluster compared to neighboring clusters.
          Scores range from -1 to 1, where values closer to 1 indicate tight, well-separated clusters and values
          near 0 suggest overlapping clusters.
        - **Accuracy Score** *(only shown at k = 2):* Since the Breast Cancer dataset has known diagnoses
          (malignant vs. benign), we can compare K-Means assignments directly against the true labels to see how
          well the algorithm recovered the real groupings — even though it never saw those labels during training.

        **Why PCA for visualization?**
        The Breast Cancer dataset has 30 features, making it impossible to plot directly. PCA compresses
        those 30 dimensions down to 2 so we can draw a scatter plot, while preserving as much of the
        original structure as possible. The cluster assignments themselves are still computed on all 30
        standardized features — PCA is only used here for the visual output.
        """)

        viz_choice_km = st.radio("Choose a visualization",
                                  ["Cluster Scatter Plot", "True Labels Comparison", "Elbow & Silhouette"],
                                  horizontal=True, key="kmeans_viz")

        if viz_choice_km == "Cluster Scatter Plot":
            st.markdown("Each point is colored by its assigned cluster. Can K-Means separate the data into meaningful groups?")
            fig, ax = plt.subplots(figsize=(8, 6))
            for cluster_label in np.unique(clusters):
                idx = np.where(clusters == cluster_label)
                ax.scatter(X_pca_km[idx, 0], X_pca_km[idx, 1],
                           alpha=0.7, edgecolor='k', s=60, label=f'Cluster {cluster_label}')
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title(f"K-Means Clustering (k={k})")
            ax.legend(loc='best')
            ax.grid(True)
            st.pyplot(fig)

        elif viz_choice_km == "True Labels Comparison":
            st.markdown("Compare the cluster assignments (left) side-by-side with the actual diagnosis labels (right).")
            col_left, col_right = st.columns(2)

            with col_left:
                fig, ax = plt.subplots(figsize=(5, 5))
                for cluster_label in np.unique(clusters):
                    idx = np.where(clusters == cluster_label)
                    ax.scatter(X_pca_km[idx, 0], X_pca_km[idx, 1],
                               alpha=0.7, edgecolor='k', s=40, label=f'Cluster {cluster_label}')
                ax.set_xlabel("PC 1")
                ax.set_ylabel("PC 2")
                ax.set_title("K-Means Clusters")
                ax.legend(loc='best')
                ax.grid(True)
                st.pyplot(fig)

            with col_right:
                colors = ['navy', 'darkorange']
                fig, ax = plt.subplots(figsize=(5, 5))
                for i, name in enumerate(target_names):
                    ax.scatter(X_pca_km[y == i, 0], X_pca_km[y == i, 1],
                               color=colors[i], alpha=0.7, edgecolor='k', s=40, label=name)
                ax.set_xlabel("PC 1")
                ax.set_ylabel("PC 2")
                ax.set_title("True Labels")
                ax.legend(loc='best')
                ax.grid(True)
                st.pyplot(fig)

            if k == 2:
                st.metric("Accuracy Score", f"{acc*100:.1f}%")
                st.markdown("*Accuracy is computed by comparing cluster assignments to the true diagnoses. "
                            "Because K-Means cluster numbers are arbitrary, we check both possible mappings "
                            "and report the higher one.*")

            st.markdown("""
            **How to read this comparison:**
            The left plot shows the groups K-Means found on its own. The right shows the actual diagnoses.
            If the two plots look similar — same rough shapes and separations — K-Means did a good job
            recovering the real structure without ever seeing the labels.

            Note that cluster numbers (0, 1) are arbitrary, so don't worry if the colors don't match
            exactly. Focus on whether the *regions* align. Points where the two plots differ are
            observations that fall in the overlapping zone between malignant and benign tumors,
            where even the algorithm finds it hard to draw a clean boundary.
            """)

        elif viz_choice_km == "Elbow & Silhouette":
            st.markdown("Run K-Means for k = 2 to 10 and inspect WCSS and silhouette score to find the best k.")
            ks = range(2, 11)
            wcss, sil_scores = [], []
            for ki in ks:
                km_i = KMeans(n_clusters=ki, random_state=42)
                km_i.fit(X_std)
                wcss.append(km_i.inertia_)
                sil_scores.append(silhouette_score(X_std, km_i.labels_))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot(ks, wcss, marker='o')
            ax1.axvline(k, color='red', linestyle='--', alpha=0.6)
            ax1.set_xlabel("Number of Clusters (k)")
            ax1.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
            ax1.set_title("Elbow Method")
            ax1.grid(True)

            ax2.plot(ks, sil_scores, marker='o', color='green')
            ax2.axvline(k, color='red', linestyle='--', alpha=0.6)
            ax2.set_xlabel("Number of Clusters (k)")
            ax2.set_ylabel("Silhouette Score")
            ax2.set_title("Silhouette Score")
            ax2.grid(True)

            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("""
            **How to interpret these plots:**

            **Elbow Method (left):** The y-axis shows the Within-Cluster Sum of Squares (WCSS) — the total
            squared distance between each point and its cluster's centroid. A lower WCSS means points are
            tighter within their clusters. As k increases, WCSS always decreases, but with diminishing returns.
            Look for the **"elbow"** — the point where the curve bends and the rate of decrease flattens out.
            The k at the elbow is typically a good choice, because adding more clusters beyond that point
            reduces WCSS only marginally while making the model unnecessarily complex.

            **Silhouette Score (right):** For each data point, the silhouette score measures how similar it is
            to its own cluster compared to the nearest other cluster. The plot shows the average score across
            all points for each k. Scores range from -1 to 1 — **higher is better**. Look for the k with the
            **highest peak**: that is where clusters are most compact and well-separated from one another.

            **Using both together:** The elbow and silhouette methods don't always agree on the same k.
            When they do, that value is a strong candidate. When they disagree, use domain knowledge to
            decide — or prefer the silhouette score, as it has a clearer interpretation.
            """)

        st.markdown("---")
        st.markdown("### Key Takeaways")
        sil_label_km = "strong" if sil > 0.5 else ("moderate" if sil > 0.25 else "weak")
        st.info(f"""
**Cluster quality:** Silhouette score = **{sil:.3f}** — {sil_label_km} cluster separation with k = {k}.
{"✅ Clusters are well-separated. K-Means recovered meaningful structure from the data." if sil > 0.5 else ("⚠️ Some points sit near cluster boundaries. The Elbow & Silhouette plot may suggest a better k." if sil > 0.25 else "❌ Clusters overlap significantly. Try a different k using the Elbow & Silhouette visualization.")}

**Inertia:** {kmeans.inertia_:,.1f} — Measures total within-cluster compactness. Use the Elbow plot to see how this changes across values of k.

{f"**Accuracy vs. true labels:** {acc*100:.1f}% — K-Means {'closely matched' if acc > 0.85 else 'partially recovered' if acc > 0.70 else 'struggled to recover'} the known malignant/benign groupings without seeing any labels." if k == 2 else "**Accuracy:** Only available at k = 2, where cluster assignments can be compared directly against the binary diagnosis labels."}

**Recommendation:** If the Elbow & Silhouette plot peaks at a different k than {k}, adjust the slider and compare. A silhouette score above 0.5 with balanced cluster sizes is a reliable sign of good grouping.
""")
        st.divider()

    else:
        st.markdown("""
        **What kind of dataset should you upload?**
        - Feature columns must be **numeric** (integers or decimals).
        - If you have a **label or target column**, keep it — you can select it to color the scatter plot.
        - Remove any **ID or index columns** — these distort distance calculations.
        - Rows with missing values in feature columns will be dropped automatically.
        """)

        upload_km = st.file_uploader("Upload a CSV file", type=["csv"], key="kmeans_upload")

        if upload_km:
            df_raw_km = pd.read_csv(upload_km)
            df_num_km = df_raw_km.select_dtypes(include=[np.number])
            valid_km = df_num_km.notna().all(axis=1)
            df_num_km = df_num_km[valid_km].reset_index(drop=True)
            df_raw_km_aligned = df_raw_km[valid_km].reset_index(drop=True)

            if df_num_km.shape[1] < 2:
                st.error("Your dataset must contain at least 2 numeric feature columns.")
            else:
                X_up_km = df_num_km.values

                st.markdown("### 2) Raw Data Preview")
                st.dataframe(df_num_km.head())
                st.divider()

                st.markdown("### 3) Standardize the Data")
                st.markdown("K-Means relies on distance calculations and can be biased by the scale of features. "
                            "Features are centered (mean = 0) and scaled (std = 1).")
                scaler_km = StandardScaler()
                X_std_km = scaler_km.fit_transform(X_up_km)
                st.dataframe(pd.DataFrame(X_std_km, columns=df_num_km.columns).head())
                st.divider()

                st.markdown("### 4) K-Means Results")

                k_up = st.slider("Number of Clusters (k)", min_value=2, max_value=10, value=2, key="upload_kmeans_k")

                km_up = KMeans(n_clusters=k_up, random_state=42)
                clusters_up = km_up.fit_predict(X_std_km)
                pca_up_km = PCA(n_components=2)
                X_pca_up_km = pca_up_km.fit_transform(X_std_km)

                sil_up = silhouette_score(X_std_km, clusters_up)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Clusters (k)", k_up)
                col_b.metric("Silhouette Score", f"{sil_up:.3f}")
                col_c.metric("Inertia", f"{km_up.inertia_:,.1f}")

                st.markdown("""
                **Understanding the metrics:**
                - **Clusters (k):** The number of groups you asked K-Means to find. There is no single correct value —
                the goal is to choose a k that produces meaningful, well-separated groups in your data.
                - **Silhouette Score:** Measures how well each point fits its own cluster compared to neighboring clusters.
                Scores range from -1 to 1, where values closer to 1 indicate tight, well-separated clusters and values
                near 0 suggest overlapping clusters.
                - **Inertia:** The sum of squared distances between each point and its cluster center. Lower inertia means
                points are tighter within their clusters — though it will always decrease as k increases, which is why
                it is best interpreted alongside the silhouette score.

                **Why PCA for visualization?**
                High-dimensional datasets can have tens or even hundreds of features, making them impossible to plot directly.
                PCA compresses those dimensions down to 2 so we can draw a scatter plot, while preserving as much of the
                original structure as possible. The cluster assignments themselves are still computed on all standardized
                features — PCA is only used here for the visual output.
                """)
                
                viz_km_up = st.radio("Choose a visualization",
                                     ["Cluster Scatter Plot", "True Labels Comparison", "Elbow & Silhouette"],
                                     horizontal=True, key="upload_kmeans_viz")

                if viz_km_up == "Cluster Scatter Plot":
                    st.markdown("Each point is colored by its assigned cluster. Can K-Means separate the data into meaningful groups?")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    for cl in np.unique(clusters_up):
                        idx = np.where(clusters_up == cl)
                        ax.scatter(X_pca_up_km[idx, 0], X_pca_up_km[idx, 1],
                                   alpha=0.7, edgecolor='k', s=60, label=f'Cluster {cl}')
                    ax.legend()
                    ax.set_xlabel("Principal Component 1")
                    ax.set_ylabel("Principal Component 2")
                    ax.set_title(f"K-Means Clustering (k={k_up})")
                    ax.grid(True)
                    st.pyplot(fig)

                elif viz_km_up == "True Labels Comparison":
                    categorical_cols = [c for c in df_raw_km_aligned.columns
                                        if df_raw_km_aligned[c].nunique() <= 20]
                    if not categorical_cols:
                        st.warning("No columns with 20 or fewer unique values found. Upload a dataset with a categorical label column to use this view.")
                    else:
                        label_col_km = st.selectbox(
                            "Select a categorical column to compare against cluster assignments:",
                            options=categorical_cols,
                            key="kmeans_true_label_col"
                        )
                        color_km = df_raw_km_aligned[label_col_km]

                        col_left, col_right = st.columns(2)
                        with col_left:
                            fig, ax = plt.subplots(figsize=(5, 5))
                            for cl in np.unique(clusters_up):
                                idx = np.where(clusters_up == cl)
                                ax.scatter(X_pca_up_km[idx, 0], X_pca_up_km[idx, 1],
                                           alpha=0.7, edgecolor='k', s=40, label=f'Cluster {cl}')
                            ax.set_xlabel("PC 1")
                            ax.set_ylabel("PC 2")
                            ax.set_title("K-Means Clusters")
                            ax.legend(loc='best')
                            ax.grid(True)
                            st.pyplot(fig)

                        with col_right:
                            fig, ax = plt.subplots(figsize=(5, 5))
                            for lbl in color_km.unique():
                                mask = color_km == lbl
                                ax.scatter(X_pca_up_km[mask, 0], X_pca_up_km[mask, 1],
                                           label=str(lbl), alpha=0.7, edgecolor='k', s=40)
                            ax.set_xlabel("PC 1")
                            ax.set_ylabel("PC 2")
                            ax.set_title(f"True Labels: {label_col_km}")
                            ax.legend(loc='best')
                            ax.grid(True)
                            st.pyplot(fig)

                        if color_km.nunique() == 2 and k_up == 2:
                            y_encoded, _ = pd.factorize(color_km)
                            acc_up = accuracy_score(y_encoded, clusters_up)
                            acc_up = max(acc_up, 1 - acc_up)
                            st.metric("Accuracy Score", f"{acc_up*100:.1f}%")
                            st.markdown("*Accuracy is computed by comparing cluster assignments to your chosen label column. "
                                        "Because K-Means cluster numbers are arbitrary, we check both possible mappings "
                                        "and report the higher one.*")
                            
                        elif color_km.nunique() != 2:
                            st.info(f"Accuracy is only shown when the label column has exactly 2 unique values. "
                                    f"'{label_col_km}' has {color_km.nunique()}.")
                        elif k_up != 2:
                            st.info("Set k = 2 to compare cluster assignments against a binary label column and see an accuracy score.")

                        st.markdown("""
                        **How to read this comparison:**
                        The left plot shows the groups K-Means found on its own. The right shows your chosen label column.
                        If the two plots look similar, K-Means recovered the real structure without ever seeing those labels.
                        Cluster numbers are arbitrary — focus on whether the *regions* align, not the colors.
                        """)

                elif viz_km_up == "Elbow & Silhouette":
                    ks_up = range(2, 11)
                    wcss_up, sil_up_scores = [], []
                    for ki in ks_up:
                        km_i = KMeans(n_clusters=ki, random_state=42)
                        km_i.fit(X_std_km)
                        wcss_up.append(km_i.inertia_)
                        sil_up_scores.append(silhouette_score(X_std_km, km_i.labels_))

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    ax1.plot(ks_up, wcss_up, marker='o')
                    ax1.axvline(k_up, color='red', linestyle='--', alpha=0.6)
                    ax1.set_xlabel("k")
                    ax1.set_ylabel("WCSS")
                    ax1.set_title("Elbow Method")
                    ax1.grid(True)
                    ax2.plot(ks_up, sil_up_scores, marker='o', color='green')
                    ax2.axvline(k_up, color='red', linestyle='--', alpha=0.6)
                    ax2.set_xlabel("k")
                    ax2.set_ylabel("Silhouette Score")
                    ax2.set_title("Silhouette Score")
                    ax2.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)

                    st.markdown("""
                    **How to interpret these plots:**

                    **Elbow Method (left):** The y-axis shows the Within-Cluster Sum of Squares (WCSS) — the total
                    squared distance between each point and its cluster's centroid. A lower WCSS means points are
                    tighter within their clusters. As k increases, WCSS always decreases, but with diminishing returns.
                    Look for the **"elbow"** — the point where the curve bends and the rate of decrease flattens out.
                    The k at the elbow is typically a good choice, because adding more clusters beyond that point
                    reduces WCSS only marginally while making the model unnecessarily complex.

                    **Silhouette Score (right):** For each data point, the silhouette score measures how similar it is
                    to its own cluster compared to the nearest other cluster. The plot shows the average score across
                    all points for each k. Scores range from -1 to 1 — **higher is better**. Look for the k with the
                    **highest peak**: that is where clusters are most compact and well-separated from one another.

                    **Using both together:** The elbow and silhouette methods don't always agree on the same k.
                    When they do, that value is a strong candidate. When they disagree, use domain knowledge to
                    decide — or prefer the silhouette score, as it has a clearer interpretation.
                    """)

                st.markdown("---")
                st.markdown("### Key Takeaways")
                sil_up_label = "strong" if sil_up > 0.5 else ("moderate" if sil_up > 0.25 else "weak")
                st.info(f"""
**Cluster quality:** Silhouette score = **{sil_up:.3f}** — {sil_up_label} cluster separation with k = {k_up}.
{"✅ Clusters are well-separated. K-Means found meaningful structure in your data." if sil_up > 0.5 else ("⚠️ Some points sit near cluster boundaries. The Elbow & Silhouette plot may suggest a better k." if sil_up > 0.25 else "❌ Clusters overlap significantly. Your data may not have a strong cluster structure, or k may need adjustment.")}

**Inertia:** {km_up.inertia_:,.1f} — Measures total within-cluster compactness. Use the Elbow plot to see how this changes across values of k.

**Recommendation:** If the Elbow & Silhouette plots suggest a different k than {k_up}, adjust the slider and re-evaluate. A silhouette score above 0.5 with balanced cluster sizes is a reliable sign of good grouping.
""")

    if st.button("← Back to Home", key="kmeans_back_bottom"):
        st.session_state.page = "home"
        st.rerun()


def show_hierarchical():
    scroll_to_top()
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("Hierarchical Clustering 🟠")
    st.markdown("### What is Hierarchical Clustering?")
    st.markdown("""
    Hierarchical clustering builds a **tree of nested clusters** (called a *dendrogram*) by
    progressively merging the most similar observations together.

    Unlike K-Means, you don't need to decide the number of clusters upfront —
    you grow the full tree first, then "cut" it at any level to produce as many groups as you want.

    **The algorithm (agglomerative / bottom-up):**
    1. Start with every observation as its own cluster.
    2. Find the two closest clusters and merge them.
    3. Repeat until everything is in one cluster.
                
    How clusters are merged depends on the linkage method chosen:            
    **Linkage methods** control how "distance between clusters" is measured:
    - **Ward** — minimizes the increase in total within-cluster variance at each merge. Usually the best starting point.
    - **Complete** — uses the maximum pairwise distance. Creates compact, roughly equal-sized clusters.
    - **Average** — uses the mean pairwise distance. A middle ground.
    - **Single** — uses the minimum pairwise distance. Can produce elongated "chaining" clusters.

    **What the dendrogram tells you:** Observations that merge early (low on the tree) are very similar.
    The height of a merge reflects how different the two clusters were when they joined. Long vertical
    branches indicate a natural break in the data — a good place to draw a horizontal line to define your clusters.
    """)

    st.markdown("### 1) Choose a Dataset")
    data_source = st.radio("Data Source", ["Sample Dataset", "Upload CSV File"], key="hc_data_source")

    # ── SAMPLE DATASET ──────────────────────────────────────────────────────────
    if data_source == "Sample Dataset":
        st.markdown("""
        You have selected the **Democracy & Dictatorship 2020** dataset.

        Each row is a country. Features describe political and electoral indicators:
        regime type, democracy scores, voting systems, and constitutional characteristics.
        The goal is to see whether hierarchical clustering can recover meaningful groupings
        of countries — purely from the structure of these indicators, with no labels.
        """)

        try:
            with st.spinner("Loading dataset…"):
                features_df, country_names, country_codes = load_democracy_data()
        except Exception:
            st.error("Could not load the dataset. Check your internet connection and try again.")
            return

        st.markdown("### 2) Raw Data Preview")
        st.dataframe(features_df.head())
        st.markdown(f"**{len(features_df)} countries** with complete data across **{features_df.shape[1]} features**.")
        st.markdown("""
**How the features were selected:**

The original dataset has 43 columns covering country metadata, electoral events, and political classifications —
most of which are either redundant, text-based, or not useful for distance calculations. To prepare it for
clustering, three steps were applied:

1. **Column selection** — 6 continuous numeric columns were kept (e.g. `regime_category_index`, `spatial_democracy`,
   `parliament_chambers`) alongside 11 binary political indicators, for 17 features total.
2. **Boolean encoding** — columns like `is_democracy`, `is_monarchy`, and `has_free_and_fair_election` are stored
   as Python True/False values. These were converted to 1 and 0 so they can participate in distance calculations.
3. **Dropping incomplete rows** — any country missing a value in one of the 17 selected columns was excluded.
   The remaining rows are the complete cases shown above.
""")
        st.divider()

        st.markdown("### 3) Standardize the Data")
        st.markdown(
            "Hierarchical clustering uses Euclidean distance. Features on larger scales would dominate the "
            "distance calculation, so we center (mean = 0) and scale (std = 1) every feature before clustering."
        )
        scaler_hc = StandardScaler()
        X_scaled_hc = scaler_hc.fit_transform(features_df.values)
        df_scaled_hc = pd.DataFrame(X_scaled_hc, columns=features_df.columns)
        st.dataframe(df_scaled_hc.head())
        st.divider()

        st.markdown("### 4) Hierarchical Clustering Results")
        st.markdown("Choose a linkage method and number of clusters. The dendrogram updates immediately — use it to guide your choice of k, then explore the other visualizations below.")

        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            linkage_method = st.selectbox(
                "Linkage Method",
                ["ward", "complete", "average", "single"],
                key="hc_linkage",
                help="Ward is the best default. Try others to see how cluster shapes change."
            )
        with col_ctrl2:
            k_hc = st.slider("Number of Clusters (k)", min_value=2, max_value=8, value=4, key="hc_k")

        # Always-visible dendrogram
        st.markdown("#### Dendrogram")
        st.markdown("""
        The dendrogram shows the full merge history — truncated to the last 30 merges for readability.
        Each leaf may represent multiple countries (count shown in parentheses).
        The **red dashed line** marks where to cut the tree to produce your chosen k.
        Longer vertical branches before a merge signal a more natural cluster boundary.
        """)
        Z_hc = scipy_linkage(X_scaled_hc, method=linkage_method)
        n_samples_hc = X_scaled_hc.shape[0]
        cut_height = (Z_hc[n_samples_hc - k_hc - 1, 2] + Z_hc[n_samples_hc - k_hc, 2]) / 2

        fig_dend, ax_dend = plt.subplots(figsize=(12, 6))
        dendrogram(Z_hc, truncate_mode='lastp', p=30, show_contracted=True, ax=ax_dend)
        ax_dend.axhline(y=cut_height, color='red', linestyle='--', alpha=0.8, label=f'Cut for k={k_hc}')
        ax_dend.set_title(f"Hierarchical Clustering Dendrogram ({linkage_method} linkage, last 30 merges)")
        ax_dend.set_xlabel("Cluster (sample count in parentheses)")
        ax_dend.set_ylabel("Distance")
        ax_dend.legend()
        plt.tight_layout()
        st.pyplot(fig_dend)
        st.divider()

        # Cluster assignments + metrics
        agg = AgglomerativeClustering(n_clusters=k_hc, linkage=linkage_method)
        cluster_labels_hc = agg.fit_predict(X_scaled_hc)

        pca_hc = PCA(n_components=2)
        X_pca_hc = pca_hc.fit_transform(X_scaled_hc)
        sil_hc = silhouette_score(X_scaled_hc, cluster_labels_hc)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Clusters (k)", k_hc)
        col_m2.metric("Silhouette Score", f"{sil_hc:.3f}")
        col_m3.metric("Countries Clustered", len(country_names))

        cluster_counts = pd.Series(cluster_labels_hc).value_counts().sort_index()
        st.markdown("**Cluster sizes:** " + " · ".join([f"Cluster {i}: {n}" for i, n in cluster_counts.items()]))

        st.markdown("""
        **Understanding the metrics:**
        - **Clusters (k):** The number of groups produced by cutting the dendrogram at the red line.
          Use the dendrogram above to check that the cut lands at a natural gap between long branches.
        - **Silhouette Score:** How well each country fits its own cluster compared to the nearest other cluster.
          Ranges from -1 to 1 — closer to 1 means tighter, better-separated clusters.
          Use the Silhouette Analysis visualization below to find the k that maximizes this score.
        - **PCA for visualization:** The 17 features are compressed to 2 dimensions only for the scatter plot.
          Clustering is computed on all 17 standardized features — PCA does not affect the groupings.
        """)

        viz_hc = st.radio(
            "Explore further",
            ["Cluster Scatter Plot (PCA)", "World Map", "Silhouette Analysis"],
            horizontal=True,
            key="hc_viz"
        )

        if viz_hc == "Cluster Scatter Plot (PCA)":
            st.markdown(
                "Each point is a country, colored by its assigned cluster. Hover to see individual countries. "
                "Points are projected onto the first 2 principal components — "
                "groups that look close here are genuinely similar across all 17 political features."
            )
            results_scatter = pd.DataFrame({
                "country": country_names,
                "PC1": X_pca_hc[:, 0],
                "PC2": X_pca_hc[:, 1],
                "Cluster": cluster_labels_hc.astype(str)
            })
            fig_scatter = px.scatter(
                results_scatter, x="PC1", y="PC2", color="Cluster",
                hover_name="country",
                title=f"Agglomerative Clustering (k={k_hc}, {linkage_method} linkage)",
                labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_scatter.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color='black')))
            fig_scatter.update_layout(legend_title_text="Cluster")
            st.plotly_chart(fig_scatter, use_container_width=True)

        elif viz_hc == "World Map":
            st.markdown(
                "Each country is shaded by its cluster assignment. Countries with similar political "
                "structures should cluster together — look for regional or geopolitical patterns."
            )
            results_map = pd.DataFrame({
                "country": country_names,
                "country_code": country_codes,
                "Cluster": cluster_labels_hc.astype(str)
            })
            fig_map = px.choropleth(
                results_map,
                locations="country_code",
                locationmode="ISO-3",
                color="Cluster",
                hover_name="country",
                title=f"Country Clusters — Democracy & Dictatorship 2020 (k={k_hc}, {linkage_method})",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_map.update_geos(fitbounds="locations", visible=True)
            fig_map.update_layout(legend_title_text="Cluster")
            st.plotly_chart(fig_map, use_container_width=True)

        elif viz_hc == "Silhouette Analysis":
            st.markdown(
                "Silhouette scores are computed for k = 2 to 8 using the same linkage method. "
                "The **red dashed line** marks your current k. "
                "The peak indicates the k that produces the most naturally separated clusters."
            )
            with st.spinner("Computing silhouette scores…"):
                k_range_hc = range(2, 9)
                sil_scores_hc = [
                    silhouette_score(X_scaled_hc,
                                     AgglomerativeClustering(n_clusters=ki, linkage=linkage_method).fit_predict(X_scaled_hc))
                    for ki in k_range_hc
                ]
            best_k_hc = list(k_range_hc)[int(np.argmax(sil_scores_hc))]

            fig_sil, ax_sil = plt.subplots(figsize=(8, 5))
            ax_sil.plot(list(k_range_hc), sil_scores_hc, marker='o', linewidth=2)
            ax_sil.axvline(k_hc, color='red', linestyle='--', alpha=0.7, label=f'Current k={k_hc}')
            ax_sil.set_xlabel("Number of Clusters (k)")
            ax_sil.set_ylabel("Silhouette Score")
            ax_sil.set_title(f"Silhouette Analysis — {linkage_method} linkage")
            ax_sil.set_xticks(list(k_range_hc))
            ax_sil.legend()
            ax_sil.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_sil)
            st.info(f"Best k by silhouette score: **{best_k_hc}** (score = {max(sil_scores_hc):.3f})")

        st.markdown("---")
        st.markdown("### Key Takeaways")
        sil_label_hc = "strong" if sil_hc > 0.5 else ("moderate" if sil_hc > 0.25 else "weak")
        st.info(f"""
**Cluster quality:** Silhouette score = **{sil_hc:.3f}** — {sil_label_hc} cluster separation.
{"✅ Clusters are well-defined and meaningfully separated." if sil_hc > 0.5 else ("⚠️ Some overlap between clusters — try a different k or linkage method." if sil_hc > 0.25 else "❌ Clusters overlap significantly. Try a different linkage method or adjust k.")}

**Linkage method used:** {linkage_method} — {"minimizes within-cluster variance; best general-purpose choice." if linkage_method == "ward" else "uses maximum pairwise distance; produces compact, equal-sized clusters." if linkage_method == "complete" else "uses mean pairwise distance; a balance between ward and single." if linkage_method == "average" else "uses minimum pairwise distance; prone to chaining — best for elongated clusters."}

**Cluster sizes:** {" · ".join([f"Cluster {i}: {n}" for i, n in pd.Series(cluster_labels_hc).value_counts().sort_index().items()])}
{"⚠️ Cluster sizes are very unbalanced — one group may be absorbing noise. Try reducing k or switching to complete linkage." if pd.Series(cluster_labels_hc).value_counts().max() / len(cluster_labels_hc) > 0.7 else "✅ Cluster sizes are reasonably balanced."}

**What to explore next:** Check the World Map to see whether the clusters align with geopolitical regions, and use the Silhouette Analysis to confirm that k = {k_hc} is the optimal cut.
""")
        st.divider()

    # ── UPLOAD CSV ───────────────────────────────────────────────────────────────
    else:
        st.markdown("Upload a CSV file. The app will automatically use all numeric columns as features.")
        uploaded_hc = st.file_uploader("Upload CSV", type=["csv"], key="hc_upload")

        if uploaded_hc is not None:
            df_raw_hc = pd.read_csv(uploaded_hc)
            num_cols_hc = df_raw_hc.select_dtypes(include=[np.number]).columns.tolist()

            if len(num_cols_hc) < 2:
                st.error("Your dataset needs at least 2 numeric columns to cluster.")
                return

            st.markdown("### 2) Raw Data Preview")
            st.dataframe(df_raw_hc.head())
            st.divider()

            st.markdown("### 3) Select Features")
            selected_cols_hc = st.multiselect(
                "Choose numeric columns to include (select at least 2):",
                options=num_cols_hc,
                default=num_cols_hc,
                key="hc_cols"
            )
            if len(selected_cols_hc) < 2:
                st.warning("Select at least 2 columns to continue.")
                return

            df_feat_hc = df_raw_hc[selected_cols_hc].dropna()
            if len(df_feat_hc) < 10:
                st.error(f"Only {len(df_feat_hc)} complete rows after dropping NaNs — need at least 10.")
                return

            st.markdown(f"**{len(df_feat_hc)} rows** used after dropping incomplete rows.")
            st.divider()

            st.markdown("### 4) Standardize the Data")
            st.markdown(
            "Hierarchical clustering uses Euclidean distance. Features on larger scales would dominate the "
            "distance calculation, so we center (mean = 0) and scale (std = 1) every feature before clustering."
        )
            scaler_up_hc = StandardScaler()
            X_scaled_up_hc = scaler_up_hc.fit_transform(df_feat_hc.values)
            df_scaled_up_hc = pd.DataFrame(X_scaled_up_hc, columns=df_feat_hc.columns)
            st.dataframe(df_scaled_up_hc.head())
            st.divider()

            st.markdown("### 5) Hierarchical Clustering Results")
            st.markdown("Choose a linkage method and number of clusters. The dendrogram updates immediately — use it to guide your choice of k, then explore the other visualizations below.")

            col_uc1, col_uc2 = st.columns(2)
            with col_uc1:
                linkage_up_hc = st.selectbox(
                    "Linkage Method",
                    ["ward", "complete", "average", "single"],
                    key="hc_up_linkage",
                    help="Ward is the best default. Try others to see how cluster shapes change."
                )
            with col_uc2:
                k_up_hc = st.slider("Number of Clusters (k)", min_value=2, max_value=min(8, len(df_feat_hc) - 1),
                                    value=3, key="hc_up_k")

            # Always-visible dendrogram
            st.markdown("#### Dendrogram")
            st.markdown(
                "Truncated to the last 30 merges. Numbers in parentheses show how many observations "
                "are in each contracted leaf. The **red dashed line** marks where to cut the tree to produce your chosen k. "
                "Longer vertical branches before a merge signal a more natural cluster boundary."
            )
            Z_up_hc = scipy_linkage(X_scaled_up_hc, method=linkage_up_hc)
            n_up = X_scaled_up_hc.shape[0]
            cut_up = (Z_up_hc[n_up - k_up_hc - 1, 2] + Z_up_hc[n_up - k_up_hc, 2]) / 2

            fig_du, ax_du = plt.subplots(figsize=(12, 6))
            p_val = min(30, n_up - 1)
            dendrogram(Z_up_hc, truncate_mode='lastp', p=p_val, show_contracted=True, ax=ax_du)
            ax_du.axhline(y=cut_up, color='red', linestyle='--', alpha=0.8, label=f'Cut for k={k_up_hc}')
            ax_du.set_title(f"Dendrogram ({linkage_up_hc} linkage)")
            ax_du.set_xlabel("Cluster (sample count in parentheses)")
            ax_du.set_ylabel("Distance")
            ax_du.legend()
            plt.tight_layout()
            st.pyplot(fig_du)
            st.divider()

            # Cluster assignments + metrics
            agg_up = AgglomerativeClustering(n_clusters=k_up_hc, linkage=linkage_up_hc)
            labels_up_hc = agg_up.fit_predict(X_scaled_up_hc)

            sil_up_hc = silhouette_score(X_scaled_up_hc, labels_up_hc)
            col_um1, col_um2, col_um3 = st.columns(3)
            col_um1.metric("Clusters (k)", k_up_hc)
            col_um2.metric("Silhouette Score", f"{sil_up_hc:.3f}")
            col_um3.metric("Rows Clustered", len(df_feat_hc))

            cluster_counts_up = pd.Series(labels_up_hc).value_counts().sort_index()
            st.markdown("**Cluster sizes:** " + " · ".join([f"Cluster {i}: {n}" for i, n in cluster_counts_up.items()]))

            st.markdown("""
            **Understanding the metrics:**
            - **Clusters (k):** The number of groups produced by cutting the dendrogram at the red line.
              Use the dendrogram above to check that the cut lands at a natural gap between long branches.
            - **Silhouette Score:** How well each observation fits its own cluster compared to the nearest other cluster.
              Ranges from -1 to 1 — closer to 1 means tighter, better-separated clusters. Use the Silhouette Analysis visualization below to find the k that maximizes this score.
            - **PCA for visualization:** Your features are compressed to 2 dimensions only for the scatter plot.
              Clustering is computed on all standardized features — PCA does not affect the groupings.
            """)

            viz_up_hc = st.radio(
                "Explore further",
                ["Cluster Scatter Plot (PCA)", "Silhouette Analysis"],
                horizontal=True,
                key="hc_up_viz"
            )

            if viz_up_hc == "Cluster Scatter Plot (PCA)":
                pca_up_hc = PCA(n_components=2)
                X_pca_up_hc = pca_up_hc.fit_transform(X_scaled_up_hc)
                ev_up_hc = pca_up_hc.explained_variance_ratio_
                st.markdown(
                    f"Points are projected onto 2 principal components capturing "
                    f"**{sum(ev_up_hc)*100:.1f}%** of the total variance."
                )
                fig_sc_up, ax_sc_up = plt.subplots(figsize=(9, 6))
                for cl in np.unique(labels_up_hc):
                    idx = labels_up_hc == cl
                    ax_sc_up.scatter(X_pca_up_hc[idx, 0], X_pca_up_hc[idx, 1],
                                     alpha=0.7, edgecolor='k', s=60, label=f'Cluster {cl}')
                ax_sc_up.set_xlabel("Principal Component 1")
                ax_sc_up.set_ylabel("Principal Component 2")
                ax_sc_up.set_title(f"Agglomerative Clustering (k={k_up_hc}, {linkage_up_hc})")
                ax_sc_up.legend()
                ax_sc_up.grid(True)
                plt.tight_layout()
                st.pyplot(fig_sc_up)

            elif viz_up_hc == "Silhouette Analysis":
                st.markdown("Silhouette scores for k = 2 to 8 with the selected linkage method.")
                with st.spinner("Computing silhouette scores…"):
                    k_range_up = range(2, min(9, len(df_feat_hc)))
                    sil_up_list = [
                        silhouette_score(X_scaled_up_hc,
                                         AgglomerativeClustering(n_clusters=ki, linkage=linkage_up_hc).fit_predict(X_scaled_up_hc))
                        for ki in k_range_up
                    ]
                best_k_up = list(k_range_up)[int(np.argmax(sil_up_list))]

                fig_su, ax_su = plt.subplots(figsize=(8, 5))
                ax_su.plot(list(k_range_up), sil_up_list, marker='o', linewidth=2)
                ax_su.axvline(k_up_hc, color='red', linestyle='--', alpha=0.7, label=f'Current k={k_up_hc}')
                ax_su.set_xlabel("Number of Clusters (k)")
                ax_su.set_ylabel("Silhouette Score")
                ax_su.set_title(f"Silhouette Analysis — {linkage_up_hc} linkage")
                ax_su.set_xticks(list(k_range_up))
                ax_su.legend()
                ax_su.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_su)
                st.info(f"Best k by silhouette score: **{best_k_up}** (score = {max(sil_up_list):.3f})")

            st.markdown("---")
            st.markdown("### Key Takeaways")
            sil_label_up_hc = "strong" if sil_up_hc > 0.5 else ("moderate" if sil_up_hc > 0.25 else "weak")
            st.info(f"""
**Cluster quality:** Silhouette score = **{sil_up_hc:.3f}** — {sil_label_up_hc} cluster separation.
{"✅ Clusters are well-defined and meaningfully separated." if sil_up_hc > 0.5 else ("⚠️ Some overlap between clusters — consider trying a different k or linkage method." if sil_up_hc > 0.25 else "❌ Clusters overlap significantly. Try a different linkage method or revisit your feature selection.")}

**Linkage method used:** {linkage_up_hc} — {"minimizes within-cluster variance; best general-purpose choice." if linkage_up_hc == "ward" else "uses maximum pairwise distance; produces compact, equal-sized clusters." if linkage_up_hc == "complete" else "uses mean pairwise distance; a balance between ward and single." if linkage_up_hc == "average" else "uses minimum pairwise distance; prone to chaining — best for elongated clusters."}

**Cluster sizes:** {" · ".join([f"Cluster {i}: {n}" for i, n in pd.Series(labels_up_hc).value_counts().sort_index().items()])}
{"⚠️ Cluster sizes are very unbalanced — one cluster may be absorbing noise. Try a different k or linkage." if pd.Series(labels_up_hc).value_counts().max() / len(labels_up_hc) > 0.7 else "✅ Cluster sizes are reasonably balanced."}

**Recommendation:** Use the Silhouette Analysis to confirm whether k = {k_up_hc} is optimal, and compare the dendrogram's branch structure against your chosen cut line.
""")
            st.divider()

    if st.button("← Back to Home", key="hc_back_bottom"):
        st.session_state.page = "home"
        st.rerun()


# Router
if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "pca":
    show_pca()
elif st.session_state.page == "kmeans":
    show_kmeans()
elif st.session_state.page == "hierarchical":
    show_hierarchical()
