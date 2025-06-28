import streamlit as st
import pandas as pd
import os
from scripts.recommender import GameRecommender

# ---------- Streamlit Config ----------
st.set_page_config(
    page_title="🎮 Game Recommender App",
    page_icon="🎮",
    layout="centered"
)

# ---------- Custom Theme ----------
st.markdown("""
    <style>
        body {
            background-color: #f0f6fc;
        }

        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        }

        .stApp {
            max-width: 100vw;
            margin: 0;
            padding: 0;
        }

        h1, h2, h3 {
            color: #1e3a8a;
        }

        .stButton>button {
            background-color: #1e3a8a;
            color: white;
            padding: 0.6em 1.2em;
            border-radius: 8px;
            font-weight: 600;
            border: none;
        }

        .stButton>button:hover {
            background-color: #2563eb;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.title("🎮 Video Game Recommender System")
st.markdown("Enter your preferences to get personalized game recommendations!")

# Load the dataset and recommender with progress indicator
@st.cache_resource
def load_recommender():
    data_path = os.path.join("data", "video_game_clean.csv")
    with st.spinner("Loading recommender system and computing similarities (can take up to 1 minute)..."):
        recommender = GameRecommender(data_path)
    st.success("Recommender system loaded!")
    return recommender

recommender = load_recommender()

# ---------- User Input ----------
st.subheader("Your Gaming Preferences")

# Game title input
all_titles = sorted(recommender.df['game_title'].unique())
selected_from_list = st.selectbox("Or pick a game from the list:", ["(None)"] + all_titles)

if selected_from_list != "(None)":
    input_title = selected_from_list

input_title = st.text_input("Enter a game title you like:", value=input_title if selected_from_list != "(None)" else "")

# Number of recommendations
top_n = st.slider("Number of recommendations:", 1, 10, 5)

if st.button("Get Recommendations"):
    if input_title.strip() == "":
        st.warning("Please enter a game title.")
    else:
        with st.spinner("Searching for recommendations..."):
            recommendations = recommender.recommend_games(input_title, top_n=top_n)
        if not recommendations.empty:
            st.subheader("🎯 Recommended Games")
            for _, game in recommendations.iterrows():
                with st.expander(f"{game['game_title']} - Rating: {game['user_rating']:.1f}"):
                    st.write(f"*Genre:* {game['genre']}")
                    st.write(f"*Platform:* {game['platform']}")
                    st.write(f"*Developer:* {game['developer']}")
                    st.write(f"*Publisher:* {game['publisher']}")
                    st.write(f"*Release Year:* {game['release_year']}")
                    st.write(f"*Price:* ${game['price']:.2f}")
                    st.write(f"*Graphics Quality:* {game['graphics_quality']}")
                    st.write(f"*Soundtrack Quality:* {game['soundtrack_quality']}")
                    st.write(f"*Story Quality:* {game['story_quality']}")
                    if not pd.isna(game['user_review_text']):
                        st.write("*Sample Review:*")
                        st.write(game['user_review_text'])
        else:
            st.warning("No recommendations found for this title. Try another game.")

# ---------- Footer ----------
st.markdown("---")
st.markdown("Made by Group 6")