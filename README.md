# Movie Recommendation System

A comprehensive movie recommendation system built using the MovieLens dataset. This project demonstrates various recommendation techniques including popularity-based filtering, content-based filtering, and collaborative filtering.

## Dataset

This project uses the [MovieLens small dataset](https://grouplens.org/datasets/movielens/) which contains approximately 100,000 ratings from 600+ users on 9,000+ movies. This smaller dataset is more manageable for version control purposes compared to the full dataset which has over 33 million ratings.

> Note: We switched from the full dataset (ml-latest) to the small dataset (ml-latest-small) for easier version control and sharing.

The dataset includes:

- `movies.csv`: Information about movies (ID, title, genres)
- `ratings.csv`: User ratings for movies (userID, movieID, rating, timestamp)
- `tags.csv`: User-generated tags for movies
- `links.csv`: Links to movie pages on IMDb and TMDb
- `genome-scores.csv` & `genome-tags.csv`: Tag relevance data

## Features

The recommendation system implements:

1. **Popularity-Based Recommendation**: Recommends movies that are highly rated by many users.
2. **Content-Based Filtering**: Recommends movies similar to ones the user liked based on movie features (genres).
3. **Collaborative Filtering**: Recommends movies based on user similarity using Matrix Factorization (SVD).

## Getting Started

### Prerequisites

For Jupyter Notebook:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn surprise jupyter
```

For Streamlit Web App:
```bash
pip install -r requirements.txt
```

### Running the Jupyter Notebook

The main analysis and recommendation algorithms are implemented in the Jupyter notebook:

```bash
jupyter notebook movie_recommendation_system.ipynb
```

### Running the Streamlit Web App

The system includes a beautiful Streamlit web app with a Gruvbox Medium theme:

```bash
streamlit run app.py
```

The app features:
- Interactive movie recommendations
- Data visualizations
- Multiple recommendation approaches
- Filtering and sorting options
- User-friendly interface

### Using the Command-Line Interface

The system also provides a command-line interface for quick recommendations:

```bash
# Get popularity-based recommendations
./recommend.py --popular --num 10

# Get content-based recommendations for a movie
./recommend.py --movie "The Matrix (1999)" --num 10

# Get collaborative filtering recommendations for a user
./recommend.py --user 1 --num 10

# Use a larger sample size for better recommendations (but slower)
./recommend.py --user 1 --sample 0.1
```

## Project Structure

- `movie_recommendation_system.ipynb`: Jupyter notebook with detailed analysis and implementation
- `recommend.py`: Command-line tool for quick recommendations
- `app.py`: Streamlit web application with Gruvbox Medium theme
- `requirements.txt`: Dependencies for the Streamlit app
- `ml-latest-small/`: Directory containing the MovieLens small dataset

## Recommendation Approaches

### 1. Popularity-Based Recommendations

Simple recommendations based on the average ratings and number of ratings. This method recommends the same movies to all users regardless of personal preferences.

### 2. Content-Based Filtering

Recommends movies similar to ones the user has liked in the past based on movie attributes. In this implementation, we use movie genres to find similar movies.

### 3. Collaborative Filtering

Makes recommendations based on the similarity between users and/or items. We've implemented:

- Matrix Factorization using Singular Value Decomposition (SVD)

## Future Improvements

- Incorporate more features for content-based filtering (actors, directors, etc.)
- Implement hybrid recommendation approaches
- Add user authentication to the web app
- Include more detailed evaluation metrics
- Optimize algorithms for handling larger datasets
- Deploy the Streamlit app to a cloud service

## Applications in Data Science

This project showcases essential data science skills:

- Data preprocessing and exploration
- Feature engineering
- Model building and evaluation
- Recommendation algorithms implementation
- Handling large datasets

It's an excellent addition to a data science portfolio, demonstrating practical applications of machine learning in a real-world recommendation scenario.
