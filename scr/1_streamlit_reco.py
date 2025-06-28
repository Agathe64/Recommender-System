import streamlit as st
import pandas as pd
import os

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

# Load the dataset
@st.cache_data
def load_data():
    data_path = os.path.join("/Users/agathecauhape/EMLyon 2024-25/Canada/Recommender System/projet/data/video_game_clean.csv")
    return pd.read_csv(data_path)

df = load_data()

# ---------- User Input ----------
st.subheader("Your Gaming Preferences")

# Genre selection
genres = sorted(df['genre'].unique())
selected_genre = st.selectbox("Select your preferred genre:", genres)

# Platform selection
platforms = sorted(df['platform'].unique())
selected_platform = st.selectbox("Select your preferred platform:", platforms)

# Age group selection
age_groups = sorted(df['age_group_targeted'].unique())
selected_age = st.selectbox("Select your age group:", age_groups)

# Price range
max_price = float(df['price'].max())
price_range = st.slider("Maximum price you're willing to pay:", 0.0, max_price, max_price/2)

# Game features
col1, col2 = st.columns(2)
with col1:
    multiplayer = st.checkbox("Must have multiplayer")
with col2:
    special_device = st.checkbox("Can require special device")

# ---------- Recommend Games ----------
if st.button("Get Recommendations"):
    # Filter games based on user preferences
    recommendations = df[
        (df['genre'] == selected_genre) &
        (df['platform'] == selected_platform) &
        (df['age_group_targeted'] == selected_age) &
        (df['price'] <= price_range)
    ]
    
    if multiplayer:
        recommendations = recommendations[recommendations['multiplayer'].str.lower() == 'yes']
    
    if not special_device:
        recommendations = recommendations[recommendations['requires_special_device'].str.lower() == 'no']
    
    # Sort by user rating and get top 5
    recommendations = recommendations.sort_values('user_rating', ascending=False).head(5)
    
    if len(recommendations) > 0:
        st.subheader("🎯 Recommended Games")
        for _, game in recommendations.iterrows():
            with st.expander(f"{game['game_title']} - Rating: {game['user_rating']:.1f}"):
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
        st.warning("No games found matching your preferences. Try adjusting your criteria.")

# ---------- Footer ----------
st.markdown("---")
st.markdown("Made with ❤️ by Group 6 · Powered by Streamlit")