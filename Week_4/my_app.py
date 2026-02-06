# my_app.py

import streamlit as st

st.title("Hello, Streamlit")

st.markdown("# Hello Streamlit!")

st.write("This is my first Streamlit app!")

### Loading our csv file
import pandas as pd

st.subheader("Exploring our Dataset")

# Load the CSV file

df = pd.read_csv("data/sample_data.csv")

st.write("Here's our data!")
st.dataframe(df)

city = st.selectbox("Select a city", df["City"].unique())
filtered_df = df[df["City"] == city]

st.write(f"People in {city}")
st.dataframe(filtered_df)

# Add bar chart
st.bar_chart(df["Salary"])

import seaborn as sns

box_plot1 = sns.boxplot(x = df["City"], y = df["Salary"])

st.pyplot(box_plot1.get_figure())