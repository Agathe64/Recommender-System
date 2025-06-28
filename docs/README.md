# Recommender System for Video Game Reviews

This repository contains our full project for building and analyzing various Recommender Systems based on a dataset of video game reviews. It includes a complete pipeline from data exploration to model deployment, as well as analysis of textual data and hybrid recommendation strategies.

---

##  Repository Structure

### `notebooks/`
Jupyter notebooks documenting each major step of the project:

- `1_setup.ipynb`: Environment setup and package installation
- `2_eda.ipynb`: Exploratory Data Analysis on the raw dataset
- `3_recosystem.ipynb`: Implementation of a basic recommender system (content-based and collaborative)
- `4_text_analysis.ipynb`: NLP processing and sentiment-based recommendation insights
- `5_hybrid_system.ipynb`: Final hybrid recommender model combining structured and textual data

###  `data/`
- `video_game_reviews.csv`: Original dataset
- `video_game_clean.csv`: Cleaned version for modeling
- `text_class.csv`: Text classification output saved from notebook for reuse

###  `scripts/`
- Python scripts reflecting the logic of each notebook
- Code formatters and linters applied using:
  - `black`
  - `isort`
  - `flake8`

Includes:
- `1_streamlit_reco.py`: Streamlit app for the basic recommender
- `2_streamlit_hybrid.py`: Streamlit app for the hybrid recommender

### `docs/`
- `README.md`: This file
- `MVP_submission.pdf`: Minimum Viable Product documentation submitted

---

## Project Summary

We explore user reviews from a video game platform to build multiple recommender system variants:

- Conduct thorough exploratory analysis
- Engineer features from textual and numerical data
- Develop and evaluate content-based and collaborative filtering approaches
- Enhance with sentiment analysis and classification
- Combine insights in a hybrid recommendation system
- Deploy interactive recommender demos via Streamlit

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Agathe64/Recommender-System.git
   cd Recommender-System
