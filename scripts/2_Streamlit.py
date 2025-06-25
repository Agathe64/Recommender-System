import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Streamlit Config ----------
st.set_page_config(
    page_title="🤖 Hybrid Game Recommender",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
    <style>
        body { background-color: #f0f6fc; }
        .main { background-color: #fff; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); }
        .stApp { max-width: 100vw; margin: 0; padding: 0; }
        h1, h2, h3 { color: #1e3a8a; }
        .stButton>button { background-color: #1e3a8a; color: white; padding: 0.6em 1.2em; border-radius: 8px; font-weight: 600; border: none; }
        .stButton>button:hover { background-color: #2563eb; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Hybrid Video Game Recommender")
st.markdown("Enter a game title or pick from the list to get AI-powered recommendations!")

@st.cache_resource
def load_hybrid_recommender():
    with st.spinner("Loading model and data (can take up to 1 minute)..."):
        # Load data
        data_path = os.path.join("/Users/agathecauhape/EMLyon 2024-25/Canada/Recommender System/projet/data/text_clean.csv")
        df = pd.read_csv(data_path)
        df['game_title_lower'] = df['game_title'].str.lower()
        # Load embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Compute review embeddings
        def get_embedding(text):
            if pd.isna(text) or text.strip() == "":
                return np.zeros(model.get_sentence_embedding_dimension())
            return model.encode(text)
        df['review_embedding'] = df['user_review_text'].apply(get_embedding)
        # Prepare features
        exclude_cols = ['game_title', 'game_title_lower', 'user_review_text', 'review_embedding']
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.difference(exclude_cols)
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = ohe.fit_transform(df[categorical_cols])
        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[numerical_cols])
        X_structured = np.hstack([X_num, X_cat])
        scaler_struct = StandardScaler()
        X_struct_scaled = scaler_struct.fit_transform(X_structured)
        embeddings_matrix = np.vstack(df['review_embedding'].values)
        X_emb_scaled = normalize(embeddings_matrix)
        X_hybrid = np.hstack([X_struct_scaled, X_emb_scaled])
        similarity_matrix = cosine_similarity(X_hybrid)
        return df, similarity_matrix

df, similarity_matrix = load_hybrid_recommender()

all_titles = sorted(df['game_title'].unique())
selected_from_list = st.selectbox("Or pick a game from the list:", ["(None)"] + all_titles)
input_title = st.text_input("Enter a game title you like:", value=selected_from_list if selected_from_list != "(None)" else "")
top_n = st.slider("Number of recommendations:", 1, 10, 5)

def recommend_games_hybrid(game_title, top_n=5):
    game_title_clean = game_title.strip().lower()
    df_reset = df.reset_index(drop=True)
    matches = df_reset[df_reset['game_title_lower'].str.contains(game_title_clean, case=False, na=False)]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return df_reset.iloc[recommended_indices][['game_title', 'genre', 'platform', 'user_rating', 'developer', 'publisher', 'release_year', 'price', 'graphics_quality', 'soundtrack_quality', 'story_quality', 'user_review_text']]

if st.button("Get Recommendations"):
    if input_title.strip() == "" or input_title == "(None)":
        st.warning("Please enter or select a game title.")
    else:
        with st.spinner("Searching for recommendations..."):
            recommendations = recommend_games_hybrid(input_title, top_n=top_n)
        if not recommendations.empty:
            st.subheader("🎯 Recommended Games")
            for _, game in recommendations.iterrows():
                with st.expander(f"{game['game_title']} - Rating: {game.get('user_rating', 'N/A')}"):
                    st.write(f"**Genre:** {game['genre']}")
                    st.write(f"**Platform:** {game['platform']}")
                    st.write(f"**Developer:** {game['developer']}")
                    st.write(f"**Publisher:** {game['publisher']}")
                    st.write(f"**Release Year:** {game['release_year']}")
                    st.write(f"**Price:** ${game['price']:.2f}")
                    st.write(f"**Graphics Quality:** {game['graphics_quality']}")
                    st.write(f"**Soundtrack Quality:** {game['soundtrack_quality']}")
                    st.write(f"**Story Quality:** {game['story_quality']}")
                    if not pd.isna(game['user_review_text']):
                        st.write("**Sample Review:**")
                        st.write(game['user_review_text'])
        else:
            st.warning("No recommendations found for this title. Try another game.")

st.markdown("---")
st.markdown("Made with ❤️ by Group 6 · Powered by Streamlit")
