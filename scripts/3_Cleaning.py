# Import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn import test_train_split

# Load the dataset

# Path to the dataset, to be changed according to your local setup
PATH = "/Users/agathecauhape/EMLyon 2024-25/Canada/Recommender System/projet/data/"

file_path = os.path.join(PATH, "video_game_clean.csv")
df = pd.read_csv(file_path)

# Clean game titles and create lowercase column for matching
df['game_title'] = df['game_title'].str.strip()
df['game_title_lower'] = df['game_title'].str.lower()

# Create a game ID from the title
df['game_id'] = pd.factorize(df['game_title'])[0]
df.set_index('game_id', inplace=True)

# Select relevant features
categorical = [
    'genre', 'platform', 'game_mode', 'age_group_targeted', 'multiplayer',
               'graphics_quality', 'soundtrack_quality','story_quality'
               ]
numerical = [
    'user_rating', 'price', 'game_length_hours'
    ]

# Preprocessing pipeline
cat_transformer = OneHotEncoder(handle_unknown='ignore')
num_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, categorical),
        ('num', num_transformer, numerical)
    ])

game_features = preprocessor.fit_transform(df).toarray()

# Feature weighting
cat_weight = 2.0
num_weight = 1.0
cat_features_len = preprocessor.transformers_[0][1].get_feature_names_out().shape[0]
game_features[:, :cat_features_len] *= cat_weight
game_features[:, cat_features_len:] *= num_weight

# Compute cosine similarity between games
similarity_matrix = cosine_similarity(game_features)

def recommend_games(game_title, top_n=5):
    game_title_clean = game_title.strip().lower()
    print("Titre recherché :", repr(game_title_clean))

    # On travaille sur une version réindexée du DataFrame
    df_reset = df.reset_index(drop=True)

    # Recherche des jeux contenant le titre
    matches = df_reset[df_reset['game_title_lower'].str.contains(game_title_clean, case=False, na=False)]

    if matches.empty:
        print(f"'{game_title}' not found in dataset.")
        return pd.DataFrame()

    # On prend le premier match (position dans df_reset)
    idx = matches.index[0]

    # On récupère les scores de similarité
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Les indices recommandés (en position, car df_reset est indexé 0..N)
    recommended_indices = [i[0] for i in sim_scores[1:top_n+1]]

    # Retour des lignes correspondantes
    return df_reset.iloc[recommended_indices][['game_title', 'genre', 'platform']]

# Example recommendation
print("Recommendations for 'The Witcher 3':")
print(recommend_games('The Witcher 3'))

def evaluate_recommendation_precision(game_title, top_k=5):
    # Get original genre of the game
    game_title_clean = game_title.strip().lower()
    df_reset = df.reset_index(drop=True)
    matches = df_reset[df_reset['game_title_lower'].str.contains(game_title_clean, case=False, na=False)]

    if matches.empty:
        print(f"'{game_title}' not found in dataset.")
        return None

    input_genre = matches.iloc[0]['genre']
    print(f"Genre of '{game_title}': {input_genre}")

    # Get recommendations
    recommendations = recommend_games(game_title, top_n=top_k)

    if recommendations.empty:
        print("No recommendations found.")
        return None

    # Compute precision@k (genre match)
    same_genre_count = (recommendations['genre'] == input_genre).sum()
    precision_at_k = same_genre_count / top_k

    print(f"Precision@{top_k}: {precision_at_k:.2f}")
    return precision_at_k

evaluate_recommendation_precision("Witcher")

def global_precision_evaluation(k=5, n_samples=20):
    df_reset = df.reset_index(drop=True)

    # Sample N distinct games
    sampled_games = df_reset['game_title'].sample(n=n_samples, random_state=42).values

    precisions = []

    for title in sampled_games:
        try:
            p = evaluate_recommendation_precision(title, top_k=k)
            if p is not None:
                precisions.append(p)
        except:
            continue

    if len(precisions) == 0:
        print("No valid evaluations.")
        return

    mean_precision = np.mean(precisions)
    std_precision = np.std(precisions)

    print(f"\n Global Evaluation on {len(precisions)} samples:")
    print(f"→ Mean Precision@{k}: {mean_precision:.2f}")
    print(f"→ Std Precision@{k}: {std_precision:.2f}")


global_precision_evaluation(k=5, n_samples=20)

def run_recommender_pipeline(query_title, top_k=5):
    print(f"🔎 Searching for: '{query_title}'")
    print("-" * 40)
    
    print("🎮 Recommendations:")
    recommendations = recommend_games(query_title, top_n=top_k)
    if not recommendations.empty:
        display(recommendations)
    else:
        print("No recommendations returned.")
    
    print("\n📏 Evaluation:")
    evaluate_recommendation_precision(query_title, top_k=top_k)

run_recommender_pipeline("Witcher", top_k=5)
