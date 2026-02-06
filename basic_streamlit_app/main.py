# main.py

# Importing libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading dataset - csv file
df = pd.read_csv("data/Oasis.csv", index_col=0)

# Setting initial layout using 2 columns
col1, col2 = st.columns(2)

with col1:
    st.write("")  # spacer
    st.image("https://oasisinet.com/wp-content/uploads/2022/05/whats-the-story-morning-glory-disc-1-4de93e4bc1c6f.jpg")
    st.write("Photo by [Oasis](https://oasisinet.com/music/whats-the-story-morning-glory/)")

with col2:
    st.markdown("# Oasis")
    st.markdown("###### Streamlit App by [Eva Romero](https://www.linkedin.com/in/evaromero-/)  🎸🤘🏻")
    st.write("This is my first Streamlit app! It provides an overview of the audio features for every Oasis song and album.")
    st.markdown("""
Oasis are a British rock band from Manchester, led by brothers Liam and Noel Gallagher. 
They defined the Britpop era of the 1990s, becoming one of the most influential rock bands of their generation.
""")



# Creating next section
st.subheader("(What's The Data) Morning Glory?")

# Filtering variables
df = df[["track_name", "album_name", "album_release_year", "duration_ms", "danceability", "energy", "loudness" , "key", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "mode", "mode_name", "key_mode", "key_name"]]

df["duration_ms"] = df["duration_ms"]/60000 # Fixing duration variable from milliseconds to minutes

df = df.rename(columns={"duration_ms": "duration_minutes"}) # Renaming duration variable

st.dataframe(df) # Displaying the dataframe

st.markdown("#### How long are the songs?")

# Creating next section using 2 columns
col3, col4 = st.columns(2)

with col3:
    plot1= sns.histplot(x = df["duration_minutes"], color = "palevioletred")
    plt.title("Distribution of track duration (minutes)")
    plt.xlabel("Duration in Minutes")
    plt.ylabel("Track frequency")
    st.pyplot(plot1.get_figure())
    plt.clf()

with col4:
    album_name = st.selectbox("Select an album", df["album_name"].unique())
    filtered_df = df[df["album_name"] == album_name]
    median_duration = filtered_df["duration_minutes"].median()
    st.write(f"The median track duration in {album_name} is {median_duration:.2f} minutes")


if st.button("**Click here for a fun fact!**"):
    st.markdown(
        "**Did you know??**\n\n"
        "The longest Oasis track is called **'Better Man'** from the *Heathen Chemistry* album.\n\n"
        "It's **38:03 minutes long**.\n\n"
        "The main song plays from **0:00–4:20**, followed by **28 minutes of silence**.\n"
        "A hidden track starts at **33:13** called *'The Cage (Instrumental)'*.\n\n"
        "This was common during the music CD era, when listeners couldn’t fast-forward \n"
        "freely or check a song’s full runtime. Discovering a hidden track meant sitting \n"
        "through long stretches of silence, wondering if the CD was broken or if the song was actually over."
    )

st.markdown("#### Audio Features")

# Creating next section using 4 columns
col5, col6, col7 = st.columns(3)

with col5:
    plot2 = sns.kdeplot(x = df["acousticness"], color= "darkslateblue", fill = True)
    plt.title("Acousticness")
    plt.xlabel("Value")
    plt.ylabel("Density")
    st.pyplot(plot2.get_figure())
    plt.clf()
    st.write("A confidence measure from 0.0 to 1.0 of whether the track is acoustic.")

with col6:
    plot3 = sns.kdeplot(x = df["danceability"], color = "mediumvioletred", fill = True)
    plt.title("Danceability")
    plt.xlabel("Value")
    plt.ylabel("Density")
    st.pyplot(plot3.get_figure())
    plt.clf()
    st.write("Describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.")

with col7:
    plot4 = sns.kdeplot(x = df["loudness"], color = "darkseagreen", fill = True)
    plt.title("Loudness")
    plt.xlabel("Value")
    plt.ylabel("Density")
    st.pyplot(plot4.get_figure())
    plt.clf()
    st.write("The overall loudness of a track in decibels (typical range between -60 and 0 db).")


col8, col9, col10 = st.columns(3)

with col8:
    plot5 = sns.kdeplot(x = df["energy"], color = "steelblue", fill = True)
    plt.title("Energy")
    plt.xlabel("Value")
    plt.ylabel("Density")
    st.pyplot(plot5.get_figure())
    plt.clf()
    st.write("Represents a perceptual measure of intensity and activity (from 0.0 to 1.0).")

with col9:
    plot7 = sns.kdeplot(x = df["speechiness"], color = "indianred", fill = True)
    plt.title("Speechiness")
    plt.xlabel("Value")
    plt.ylabel("Density")
    st.pyplot(plot7.get_figure())
    plt.clf()
    st.write("Detects the presence of spoken words in a track. The more exclusively speech-like the recording, the closer to 1.0 the attribute value.")

with col10:
    plot6 = sns.kdeplot(x = df["instrumentalness"], color = "teal", fill = True)
    plt.title("Instrumentalness")
    plt.xlabel("Value")
    plt.ylabel("Density")
    st.pyplot(plot6.get_figure())
    plt.clf()
    st.write("Predicts whether a track contains no vocals. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content.")

# Creating next section
st.subheader("Bonus!")
st.write("I bet you're wondering (probably not), but I'll tell you anyways.")
st.markdown("Here's a list of **MY FAVORITE OASIS SONGS** 🎧")


st.markdown("""
1. [Cast No Shadow](https://www.youtube.com/watch?v=p28UH0fY-7Y)
2. [Wonderwall](https://www.youtube.com/watch?v=6hzrDeceEKc)
3. [Don’t Look Back in Anger](https://www.youtube.com/watch?v=r8OipmKFDeM)
4. [Champagne Supernova](https://www.youtube.com/watch?v=tI-5uv4wryI)
5. [Stand by Me](https://youtu.be/maTP315XZCQ?si=unqaQI27qx9IkbSz)
6. [Stop Crying Your Heart Out](https://www.youtube.com/watch?v=dhZUsNJ-LQU)
7. [Supersonic](https://www.youtube.com/watch?v=BJKpUH2kJQg)
8. [Live Forever](https://www.youtube.com/watch?v=TDe1DqxwJoc)
9. [Married With Children](https://www.youtube.com/watch?v=EOoidN2s5qY)
10. [Some Might Say](https://www.youtube.com/watch?v=oxOjIj18I9E)
""")

st.markdown("***")
st.markdown("Data provided by [Oasis discography](https://www.kaggle.com/datasets/federicoseijo/oasis-discography/data)")

# streamlit run main.py

