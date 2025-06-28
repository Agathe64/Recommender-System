# Libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# Load and display the dataset

# Path to the dataset, to be changed according to your local setup
PATH = "/Users/agathecauhape/EMLyon 2024-25/Canada/Recommender System/projet/data/"

file_path = os.path.join(PATH, "video_game_clean.csv")
df = pd.read_csv(file_path)
print(df.head(3))

print("Shape of dataset:", df.shape)
print("\nInfo:")
df.info()

print("\nSummary statistics (numeric columns):")
print(df.describe())


# Numeric and Categorical Columns Detection
df_numeric = df.select_dtypes(include=['int', 'float'])
df_categorical = df.select_dtypes(include=['object', 'category', 'bool'])

print("\nNumeric columns:", df_numeric.columns.tolist())
print("Categorical columns:", df_categorical.columns.tolist())

# Distribution of Numeric Columns
for col in df_numeric.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Boxplots for Outlier Detection
for col in df_numeric.columns:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()

# Distribution
for col in df_categorical.columns:
    plt.figure(figsize=(10, 4))
    order = df[col].value_counts().index[:10]  # top 10
    sns.countplot(data=df, x=col, order=order)
    plt.title(f"Top 10 Most Common Values in {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Correlation Matrix of Numeric Features
plt.figure(figsize=(10, 8))
corr = df_numeric.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Pairplot for selected numeric features
if len(df_numeric.columns) <= 5:
    sns.pairplot(df_numeric)
    plt.suptitle("Pairplot of Numeric Features", y=1.02)
    plt.show()

# Boxplots of Numeric by Top Categorical Columns
top_cat_cols = [col for col in df_categorical.columns if df[col].nunique() < 10]

for cat_col in top_cat_cols:
    for num_col in df_numeric.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=cat_col, y=num_col, data=df)
        plt.title(f"{num_col} by {cat_col}")
        plt.tight_layout()
        plt.show()

# Text Length
if 'user_review_text' in df.columns:
    df['review_length'] = df['user_review_text'].fillna('').apply(len)
    plt.figure(figsize=(10, 4))
    sns.histplot(df['review_length'], bins=30, kde=True)
    plt.title("Distribution of Review Text Length")
    plt.tight_layout()
    plt.show()

