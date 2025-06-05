#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Movie Recommendation System - CLI Tool

This script provides a command-line interface for the movie recommendation system.
It allows users to get movie recommendations based on user ID, movie title, or just popular movies.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD

# File paths
MOVIELENS_PATH = 'ml-latest-small/'
MOVIES_FILE = os.path.join(MOVIELENS_PATH, 'movies.csv')
RATINGS_FILE = os.path.join(MOVIELENS_PATH, 'ratings.csv')

def load_data():
    """Load and prepare the MovieLens small dataset"""
    print("Loading data...")
    
    # Load movies and ratings
    movies_df = pd.read_csv(MOVIES_FILE)
    ratings_df = pd.read_csv(RATINGS_FILE)
    print(f"Loaded {len(movies_df):,} movies and {len(ratings_df):,} ratings")
    
    # Since we're using the small dataset, no need to sample
    ratings_sample = ratings_df
    
    return movies_df, ratings_df, ratings_sample

def build_popularity_model(movies_df, ratings_df, min_ratings=20):
    """Build a popularity-based recommendation model"""
    # Calculate average rating and number of ratings for each movie
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    
    movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
    
    # Merge with movie titles
    movie_stats = movie_stats.merge(movies_df[['movieId', 'title']], on='movieId')
    
    # Filter to consider only movies with a significant number of ratings
    popular_movies = movie_stats[movie_stats['num_ratings'] > min_ratings].sort_values('avg_rating', ascending=False)
    
    return popular_movies

def build_content_based_model(movies_df):
    """Build a content-based recommendation model using movie genres"""
    print("Building content-based model...")
    
    # Create TF-IDF vectors based on genres
    movies_df['genres'] = movies_df['genres'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    
    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Create a mapping from movie title to index
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
    
    return cosine_sim, indices

def get_content_based_recommendations(title, movies_df, cosine_sim, indices, n=10):
    """Get content-based recommendations for a movie"""
    # Get the index of the movie that matches the title
    try:
        idx = indices[title]
    except KeyError:
        print(f"Movie '{title}' not found in the database.")
        return pd.DataFrame(columns=['title', 'genres'])
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the N most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:n+1]
    
    # Get the movie indices and similarity scores
    movie_indices = [i[0] for i in sim_scores]
    similarities = [i[1] for i in sim_scores]
    
    # Create a dataframe with the recommended movies
    recommendations = movies_df.iloc[movie_indices].copy()
    recommendations['similarity_score'] = similarities
    
    return recommendations[['title', 'genres', 'similarity_score']]

def build_collaborative_filtering_model(ratings_sample):
    """Build a collaborative filtering model using SVD"""
    print("Building collaborative filtering model...")
    
    # Prepare the data for Surprise library
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_sample[['userId', 'movieId', 'rating']], reader)
    
    # Build the full training set
    trainset = data.build_full_trainset()
    
    # Use SVD algorithm
    svd = SVD(n_factors=100, n_epochs=20, random_state=42)
    svd.fit(trainset)
    
    return svd

def get_collaborative_filtering_recommendations(user_id, svd, movies_df, ratings_df, n=10):
    """Get collaborative filtering recommendations for a user"""
    # Check if user exists in the dataset
    if user_id not in ratings_df['userId'].values:
        print(f"User {user_id} not found in the dataset.")
        return pd.DataFrame(columns=['title', 'genres', 'predicted_rating'])
    
    # Get all movies the user hasn't rated
    user_rated_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])
    all_movies = set(movies_df['movieId'])
    unrated_movies = list(all_movies - user_rated_movies)
    
    # If there are too many movies, take a sample to speed up prediction
    if len(unrated_movies) > 5000:
        np.random.seed(42)
        unrated_movies = np.random.choice(list(unrated_movies), 5000, replace=False)
    
    # Predict ratings for unrated movies
    predictions = []
    for movie_id in unrated_movies:
        predicted_rating = svd.predict(user_id, movie_id).est
        predictions.append((movie_id, predicted_rating))
    
    # Sort by predicted rating and get top N
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    
    # Get movie details
    recommended_movies = []
    for movie_id, rating in top_n:
        movie_info = movies_df[movies_df['movieId'] == movie_id][['title', 'genres']].iloc[0]
        recommended_movies.append({
            'movieId': movie_id,
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'predicted_rating': rating
        })
    
    return pd.DataFrame(recommended_movies)

def main():
    """Main function to handle the command-line interface"""
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    
    # Define command-line arguments
    parser.add_argument('--user', type=int, help='User ID for collaborative filtering recommendations')
    parser.add_argument('--movie', type=str, help='Movie title for content-based recommendations')
    parser.add_argument('--popular', action='store_true', help='Get popular movie recommendations')
    parser.add_argument('--num', type=int, default=10, help='Number of recommendations to show (default: 10)')
    
    args = parser.parse_args()
    
    # Load data
    movies_df, ratings_df, ratings_sample = load_data()
    
    if args.user:
        # Collaborative filtering based on user ID
        svd = build_collaborative_filtering_model(ratings_sample)
        recommendations = get_collaborative_filtering_recommendations(args.user, svd, movies_df, ratings_df, args.num)
        if not recommendations.empty:
            print(f"\nTop {args.num} recommendations for user {args.user}:")
            print(recommendations[['title', 'genres', 'predicted_rating']].to_string(index=False))
    
    elif args.movie:
        # Content-based recommendations based on movie title
        cosine_sim, indices = build_content_based_model(movies_df)
        recommendations = get_content_based_recommendations(args.movie, movies_df, cosine_sim, indices, args.num)
        if not recommendations.empty:
            print(f"\nTop {args.num} recommendations similar to '{args.movie}':")
            print(recommendations[['title', 'genres', 'similarity_score']].to_string(index=False))
    
    elif args.popular or (not args.user and not args.movie):
        # Popularity-based recommendations
        popular_movies = build_popularity_model(movies_df, ratings_df)
        print(f"\nTop {args.num} popular movies:")
        print(popular_movies[['title', 'avg_rating', 'num_ratings']].head(args.num).to_string(index=False))
    
    print("\nDone!")

if __name__ == "__main__":
    main()
