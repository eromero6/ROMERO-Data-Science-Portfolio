# 🎸 Oasis Audio Features Dashboard

This project is an interactive **Streamlit dashboard** that explores the audio features, track lengths, and albums of the British rock band **Oasis**.  

It combines data visualization with music history to provide a fun, intuitive way to explore Oasis’s discography.

The app was built as a first Streamlit project and focuses on clean layout, clear explanations, and interactive exploration.

---

## 📊 What the App Does

The dashboard allows users to:

- View the full Oasis discography in an interactive data table
- Explore the distribution of song durations
- Select an album to view its median track length
- Learn a fun fact about Oasis’s longest track
- Visualize Spotify-style audio features using KDE plots, including:
  - Acousticness
  - Danceability
  - Loudness
  - Energy
  - Speechiness
  - Instrumentalness
- Browse a curated list of favorite Oasis songs with direct YouTube links

---

## 🧠 Data Source

The data comes from the **Oasis Discography dataset** available on Kaggle:

- https://www.kaggle.com/datasets/federicoseijo/oasis-discography

The dataset includes track-level audio features similar to those provided by Spotify’s audio analysis API.

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **Seaborn**
- **Matplotlib**

---

## 📁 Project Structure

```text
basic_streamlit_app/
├── main.py
├── data/
│   └── Oasis.csv
├── README.md
