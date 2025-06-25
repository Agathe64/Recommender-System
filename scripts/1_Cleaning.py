# Libraries for data manipulation
import pandas as pd
import os

# Load and display the dataset

# Path to the dataset, to be changed according to your local setup
PATH = "/Users/agathecauhape/EMLyon 2024-25/Canada/Recommender System/projet/data/"

file_path = os.path.join(PATH, "video_game_reviews.csv")
df = pd.read_csv(file_path)
print(df.head(3))

df.info()

# Save df into df_copy for further processing and not to modify the original df
df_copy = df.copy()

df_copy.isna().sum()

# Count duplicates
duplicate_count = df_copy.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# Get percentage of duplicates
duplicate_percentage = (duplicate_count / len(df_copy)) * 100
print(f"Percentage of duplicates: {duplicate_percentage:.2f}%")

# Change the column names
df_copy.columns = [
    'game_title',
    'user_rating',
    'age_group_targeted',
    'price',
    'platform',
    'requires_special_device',
    'developer',
    'publisher',
    'release_year',
    'genre',
    'multiplayer',
    'game_length_hours',
    'graphics_quality',
    'soundtrack_quality',
    'story_quality',
    'user_review_text',
    'game_mode',
    'min_number_of_players'
]

# Check the new column names
df_copy.info()

# Fix the types of the columns
df_copy['game_title'] = df_copy['game_title'].astype('string')
df_copy['user_rating'] = df_copy['user_rating'].astype('float')
df_copy['age_group_targeted'] = df_copy['age_group_targeted'].astype('category')
df_copy['price'] = df_copy['price'].astype('float')
df_copy['platform'] = df_copy['platform'].astype('category')
df_copy['requires_special_device'] = df_copy['requires_special_device'].astype('string')
df_copy['developer'] = df_copy['developer'].astype('string')
df_copy['publisher'] = df_copy['publisher'].astype('string')
df_copy['release_year'] = df_copy['release_year'].astype('int')
df_copy['genre'] = df_copy['genre'].astype('category')
df_copy['multiplayer'] = df_copy['multiplayer'].astype('string')
df_copy['game_length_hours'] = df_copy['game_length_hours'].astype('float')
df_copy['graphics_quality'] = df_copy['graphics_quality'].astype('category')
df_copy['soundtrack_quality'] = df_copy['soundtrack_quality'].astype('category')
df_copy['story_quality'] = df_copy['story_quality'].astype('category')
df_copy['user_review_text'] = df_copy['user_review_text'].astype('string')
df_copy['game_mode'] = df_copy['game_mode'].astype('string')
df_copy['min_number_of_players'] = df_copy['min_number_of_players'].astype('int')

df_copy.describe()

# Display the count of unique values for each colmn
print(" Unique value counts per column:\n")
for col in df_copy.columns:
    print(f"{col}: {df_copy[col].nunique()} unique values")

# Verify the changes made to the DataFrame before saving
df_copy.head(3)

# Save
df_copy.to_csv(os.path.join(PATH, "video_game_clean.csv"), index=False)