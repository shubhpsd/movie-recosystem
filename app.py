#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Movie Recommendation         /* Tab list and individual tabs styling */
        div[data-testid="stHorizontalTabs"] {
            padding: 8px 8px 0px 8px !important;
            gap: 8px !important;
            background-color: {COLORS['bg_h']};
            border-radius: 8px 8px 0 0;
        }
        button[data-baseweb="tab"] {
            color: {COLORS['fg']};
            border-color: {COLORS['bg_s']};
            padding: 10px 24px !important;
            margin: 0px 5px !important;
            border-radius: 6px 6px 0 0;
            position: relative;
            transition: all 0.2s ease-in-out;
        }
        /* Selected tab styling */
        button[data-baseweb="tab"][aria-selected="true"] {
            color: {COLORS['yellow']} !important;
            background-color: {COLORS['bg']} !important;
            border-bottom: 3px solid {COLORS['yellow']} !important;
            font-weight: 500 !important;
        }
        
        /* Brute force approach for tab padding */
        div[data-testid="stHorizontalTabs"] button[role="tab"] div p {
            padding-left: 15px !important;
            padding-right: 15px !important;
            margin-left: 10px !important;
            margin-right: 10px !important;
        }
        
        /* Hover effect for tabs */
        button[data-baseweb="tab"]:hover {
            color: {COLORS['yellow']};
            background-color: {COLORS['bg_s']};
            cursor: pointer;
        }
        
        /* Text styling inside tabs */
        button[data-baseweb="tab"] div {
            font-size: 1rem;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }reamlit Web App

This Streamlit app provides a web interface for the movie recommendation system.
It allows users to get movie recommendations based on user ID, movie title, or popular movies.
The app uses a Gruvbox Medium theme for styling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
import base64
from io import BytesIO
import time

# Set paths
MOVIELENS_PATH = 'ml-latest-small/'
MOVIES_FILE = os.path.join(MOVIELENS_PATH, 'movies.csv')
RATINGS_FILE = os.path.join(MOVIELENS_PATH, 'ratings.csv')

# Gruvbox Medium theme colors
COLORS = {
    'bg_h': '#282828',       # Background hard
    'bg': '#32302f',         # Background
    'bg_s': '#504945',       # Background soft
    'fg': '#ebdbb2',         # Foreground
    'red': '#fb4934',        # Red
    'green': '#b8bb26',      # Green
    'yellow': '#fabd2f',     # Yellow
    'blue': '#83a598',       # Blue
    'purple': '#d3869b',     # Purple
    'aqua': '#8ec07c',       # Aqua/Cyan
    'orange': '#fe8019'      # Orange
}

# Custom CSS with Gruvbox Medium theme
def set_gruvbox_theme():
    st.markdown(f"""
    <style>
        body {{
            color: {COLORS['fg']};
            background-color: {COLORS['bg']};
        }}
        .stApp {{
            background-color: {COLORS['bg']};
        }}
        .stTextInput > div > div > input, .stSelectbox > div > div > input {{
            color: {COLORS['fg']};
            background-color: {COLORS['bg_s']};
            border-color: {COLORS['bg_s']};
        }}
        .stSlider > div > div > div > div {{
            background-color: {COLORS['orange']};
        }}
        .stSlider > div > div > div {{
            background-color: {COLORS['bg_s']};
        }}
        h1, h2, h3 {{
            color: {COLORS['yellow']};
        }}
        h4, h5, h6 {{
            color: {COLORS['aqua']};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {COLORS['bg_h']};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {COLORS['fg']};
            border-color: {COLORS['bg_s']};
        }}
        .stTabs [aria-selected="true"] {{
            color: {COLORS['yellow']};
            border-bottom-color: {COLORS['yellow']};
        }}
        .stButton > button {{
            background-color: {COLORS['bg_s']};
            color: {COLORS['fg']};
            border: 1px solid {COLORS['bg_h']};
        }}
        .stButton > button:hover {{
            background-color: {COLORS['bg_h']};
            color: {COLORS['yellow']};
            border: 1px solid {COLORS['yellow']};
        }}
        div[data-testid="stDecoration"] {{
            background-image: linear-gradient(90deg, {COLORS['red']}, {COLORS['orange']}, {COLORS['yellow']});
        }}
        .stDataFrame {{
            background-color: {COLORS['bg_s']};
        }}
        .stTable {{
            background-color: {COLORS['bg_s']};
        }}
    </style>
    """, unsafe_allow_html=True)

# Function to create custom plots with Gruvbox theme
def set_gruvbox_plot_style():
    plt.style.use('dark_background')
    sns.set_style("darkgrid")
    plt.rcParams['axes.facecolor'] = COLORS['bg']
    plt.rcParams['figure.facecolor'] = COLORS['bg']
    plt.rcParams['text.color'] = COLORS['fg']
    plt.rcParams['axes.labelcolor'] = COLORS['fg']
    plt.rcParams['xtick.color'] = COLORS['fg']
    plt.rcParams['ytick.color'] = COLORS['fg']
    plt.rcParams['grid.color'] = COLORS['bg_s']
    plt.rcParams['axes.edgecolor'] = COLORS['bg_s']
    
# Convert matplotlib figure to image for Streamlit
def plot_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=COLORS['bg'])
    buf.seek(0)
    return buf

@st.cache_data
def load_data():
    """Load and prepare the MovieLens small dataset with caching"""
    # Load movies and ratings
    movies_df = pd.read_csv(MOVIES_FILE)
    ratings_df = pd.read_csv(RATINGS_FILE)
    
    # Since we're using the small dataset, no need to sample
    ratings_sample = ratings_df
    
    return movies_df, ratings_df, ratings_sample

@st.cache_data
def build_popularity_model(movies_df, ratings_df, min_ratings=20):
    """Build a popularity-based recommendation model"""
    # Calculate average rating and number of ratings for each movie
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    
    movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
    
    # Merge with movie titles
    movie_stats = movie_stats.merge(movies_df[['movieId', 'title', 'genres']], on='movieId')
    
    # Filter to consider only movies with a significant number of ratings
    popular_movies = movie_stats[movie_stats['num_ratings'] > min_ratings].sort_values('avg_rating', ascending=False)
    
    return popular_movies

@st.cache_resource
def build_content_based_model(movies_df, num_sample=5000):
    """Build a content-based recommendation model using movie genres"""
    # First, create a soup of genres for each movie
    movies_df['genres'] = movies_df['genres'].fillna('')
    
    # Use a sample for large datasets
    if len(movies_df) > num_sample:
        # For better recommendations, include popular movies in the sample
        movie_ratings_count = pd.read_csv(RATINGS_FILE, usecols=['movieId'])
        movie_ratings_count = movie_ratings_count['movieId'].value_counts().reset_index()
        movie_ratings_count.columns = ['movieId', 'count']
        
        # Top 1000 most rated movies
        top_movies = movie_ratings_count.sort_values('count', ascending=False).head(1000)['movieId'].values
        
        # Filter movies that are in top_movies
        top_indices = movies_df[movies_df['movieId'].isin(top_movies)].index
        
        # Remaining indices to randomly sample from
        remaining_indices = movies_df[~movies_df['movieId'].isin(top_movies)].index
        random_indices = np.random.choice(remaining_indices, min(num_sample - len(top_indices), len(remaining_indices)), replace=False)
        
        # Combine both sets of indices
        sample_indices = np.concatenate([top_indices, random_indices])
        movies_sample = movies_df.iloc[sample_indices].copy()
    else:
        movies_sample = movies_df
    
    # Reset the index to make sure indices match between dataframe and similarity matrix
    movies_sample = movies_sample.reset_index(drop=True)
    
    # Create TF-IDF vectors based on genres
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_sample['genres'])
    
    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Create a dataframe mapping movie indices to movie titles
    # Use the new reset indices which will match with cosine_sim matrix
    indices = pd.Series(movies_sample.index, index=movies_sample['title']).drop_duplicates()
    
    return cosine_sim, indices, movies_sample

def get_content_recommendations(title, cosine_sim, indices, movies_sample, n=10):
    """Get content-based recommendations for a movie"""
    # Get the index of the movie that matches the title
    try:
        # First check if the title exists in the indices
        if title not in indices:
            return None
            
        # Get the position in the cosine_sim matrix
        idx = indices[title]
        
        # Verify the index is within bounds of the cosine similarity matrix
        if idx >= cosine_sim.shape[0]:
            # If out of bounds, return None
            return None
            
    except (KeyError, IndexError) as e:
        # Handle any errors
        return None
    
    try:
        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the N most similar movies (excluding the movie itself)
        sim_scores = sim_scores[1:n+1]
        
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Verify all indices are within the range of movies_sample
        valid_indices = [i for i in movie_indices if i < len(movies_sample)]
        if not valid_indices:
            return None
            
        # Create a dataframe with the recommended movies
        recommendations = movies_sample.iloc[valid_indices].copy()
        
        # Match up the similarity scores with the valid indices
        valid_scores = [score for idx, (i, score) in enumerate(sim_scores) if i in valid_indices]
        recommendations['similarity_score'] = valid_scores
        
        return recommendations[['title', 'genres', 'similarity_score']]
    except Exception as e:
        # Catch any other exceptions
        return None

@st.cache_resource
def build_collaborative_filtering_model(ratings_sample):
    """Build a collaborative filtering model using SVD"""
    # Prepare the data for Surprise library
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_sample[['userId', 'movieId', 'rating']], reader)
    
    # Build the full training set
    trainset = data.build_full_trainset()
    
    # Use SVD algorithm
    svd = SVD(n_factors=100, n_epochs=20, random_state=42)
    svd.fit(trainset)
    
    return svd

def get_cf_recommendations(user_id, svd, movies_df, ratings_sample, n=10):
    """Get collaborative filtering recommendations for a user"""
    # Check if user exists in the dataset
    if user_id not in ratings_sample['userId'].values:
        return None
    
    # Get all movies the user hasn't rated
    user_rated_movies = set(ratings_sample[ratings_sample['userId'] == user_id]['movieId'])
    
    # Use the top 2000 most popular movies for efficiency
    # This avoids having to predict ratings for tens of thousands of movies
    movie_count = ratings_sample['movieId'].value_counts()
    top_movies = set(movie_count.nlargest(2000).index)
    
    unrated_movies = list(top_movies - user_rated_movies)
    
    # Predict ratings for unrated movies
    predictions = []
    for movie_id in unrated_movies:
        if movie_id in movies_df['movieId'].values:
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

def get_user_based_recommendations(user_ratings, svd, movies_df, ratings_sample, n=10):
    """Get collaborative filtering recommendations based on user input ratings"""
    # Create a temporary user ID that doesn't exist in the dataset
    temp_user_id = int(ratings_sample['userId'].max()) + 1
    
    # Create a list to store predictions
    predictions = []
    
    # Get all movies minus the ones the user has rated
    user_rated_movies = set([rating['movieId'] for rating in user_ratings])
    
    # Use the top most popular movies for efficiency
    movie_count = ratings_sample['movieId'].value_counts()
    top_movies = set(movie_count.nlargest(2000).index)
    
    unrated_movies = list(top_movies - user_rated_movies)
    
    # Predict ratings for unrated movies
    for movie_id in unrated_movies:
        if movie_id in movies_df['movieId'].values:
            # We need to predict as if the temporary user had given the ratings
            predicted_rating = svd.predict(temp_user_id, movie_id).est
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

@st.cache_resource
def build_collaborative_filtering_model_with_user_input(ratings_sample, user_ratings=None):
    """Build a collaborative filtering model using SVD, optionally including user input ratings"""
    # Prepare the data for Surprise library
    reader = Reader(rating_scale=(0.5, 5))
    
    # Create a copy of the ratings_sample dataframe
    ratings_data = ratings_sample.copy()
    
    if user_ratings and len(user_ratings) > 0:
        # Create a temporary user ID that doesn't exist in the dataset
        temp_user_id = int(ratings_sample['userId'].max()) + 1
        
        # Create dataframe with the new user's ratings
        new_user_ratings = pd.DataFrame([
            {'userId': temp_user_id, 'movieId': rating['movieId'], 'rating': rating['rating']} 
            for rating in user_ratings
        ])
        
        # Append the new user's ratings to the original ratings
        ratings_data = pd.concat([ratings_data, new_user_ratings])
    
    # Load data into Surprise format
    data = Dataset.load_from_df(ratings_data[['userId', 'movieId', 'rating']], reader)
    
    # Build the full training set
    trainset = data.build_full_trainset()
    
    # Use SVD algorithm
    svd = SVD(n_factors=100, n_epochs=20, random_state=42)
    svd.fit(trainset)
    
    return svd

def plot_rating_distribution(ratings_sample):
    """Plot the distribution of ratings"""
    set_gruvbox_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(ratings_sample['rating'], bins=9, kde=True, color=COLORS['yellow'], ax=ax)
    ax.set_title('Distribution of Movie Ratings', color=COLORS['aqua'], fontsize=16)
    ax.set_xlabel('Rating', color=COLORS['fg'])
    ax.set_ylabel('Count', color=COLORS['fg'])
    
    return fig

def plot_genres_distribution(movies_df):
    """Plot distribution of movie genres"""
    set_gruvbox_plot_style()
    
    # Extract genres
    genres = movies_df['genres'].str.split('|').explode()
    genre_counts = genres.value_counts().head(15)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = sns.barplot(x=genre_counts.index, y=genre_counts.values, palette=[COLORS['blue'], COLORS['aqua'], COLORS['green'], COLORS['yellow'], COLORS['orange'], COLORS['red'], COLORS['purple']] * 3, ax=ax)
    
    ax.set_title('Top 15 Movie Genres', color=COLORS['aqua'], fontsize=16)
    ax.set_xlabel('Genre', color=COLORS['fg'])
    ax.set_ylabel('Count', color=COLORS['fg'])
    plt.xticks(rotation=45, color=COLORS['fg'])
    plt.tight_layout()
    
    return fig

def main():
    """Main function for the Streamlit app"""
    # Set the Gruvbox theme
    set_gruvbox_theme()
    
    # Additional styling fix for tab spacing
    st.markdown("""
    <style>
    /* Direct fix for tab padding */
    button[data-baseweb="tab"] div p {
        margin: 0 15px !important;
        padding: 5px 10px !important;
        font-size: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App title and description
    st.title("🍿 Movie Recommendation System")
    st.markdown("""
    <p style='color: {}'>This app recommends movies based on different techniques: popularity, content similarity, and collaborative filtering.</p>
    """.format(COLORS['fg']), unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Data loading options
    min_ratings = st.sidebar.slider("Minimum Number of Ratings for Movies to be considered in popularity recommendations", min_value=10, max_value=200, value=50, step=5)
    
    # Load the datasets with progress indicator
    with st.spinner("Loading data... This may take a moment."):
        movies_df, ratings_df, ratings_sample = load_data()
        
    # Display basic dataset info in sidebar
    with st.sidebar.expander("Dataset Information"):
        st.write(f"Movies: {len(movies_df):,}")
        st.write(f"Total Ratings: {len(ratings_df):,}")
        st.write(f"Unique Users: {ratings_df['userId'].nunique():,}")
    
    # Build models for all recommendation techniques
    popularity_model = build_popularity_model(movies_df, ratings_sample, min_ratings)
    
    # Extract all unique genres for filtering
    all_genres = movies_df['genres'].str.split('|').explode().dropna().unique()
    
    # Create a container for custom tabs with better spacing
    st.markdown('<br>', unsafe_allow_html=True)
    
    # Create tabs with much more clearly padded labels to ensure spacing
    tab_labels = ["📈         Analytics         ", 
                 "🔥         Popular Movies         ", 
                 "🎭         Content-Based         ", 
                 "👥         Collaborative         "]
    tabs = st.tabs(tab_labels)
    
    # Tab 1: Analytics
    with tabs[0]:
        st.header("Dataset Analytics")
        
        st.subheader("Rating Distribution")
        rating_fig = plot_rating_distribution(ratings_sample)
        st.pyplot(rating_fig)
        
        st.subheader("Movie Genres")
        genres_fig = plot_genres_distribution(movies_df)
        st.pyplot(genres_fig)
        
        # Display top rated movies as a table
        st.subheader("Top Rated Movies (with at least {} ratings)".format(min_ratings))
        top_movies = popularity_model.head(10)
        st.dataframe(top_movies[['title', 'genres', 'avg_rating', 'num_ratings']], use_container_width=True)
    
    # Tab 2: Popularity-based recommendations
    with tabs[1]:
        st.header("Popular Movies Recommendation")
        st.markdown("""
        <p style='color: {}'>This approach recommends movies that are highly rated by many users.</p>
        """.format(COLORS['fg']), unsafe_allow_html=True)
        
        # Controls for popularity recommendations
        col1, col2 = st.columns(2)
        with col1:
            pop_sort = st.selectbox("Sort by", ['Average Rating', 'Number of Ratings', 'Weighted Rating'], key='pop_sort')
        with col2:
            genre_filter = st.selectbox("Filter by genre", ['All'] + sorted(all_genres.tolist()), key='pop_genre')
        
        # Apply filters and sorting
        filtered_movies = popularity_model.copy()
        if genre_filter != 'All':
            filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(genre_filter)]
        
        if pop_sort == 'Average Rating':
            filtered_movies = filtered_movies.sort_values('avg_rating', ascending=False)
        elif pop_sort == 'Number of Ratings':
            filtered_movies = filtered_movies.sort_values('num_ratings', ascending=False)
        else:  # Weighted Rating (using IMDB formula)
            C = popularity_model['avg_rating'].mean()
            m = min_ratings
            filtered_movies['weighted_rating'] = (filtered_movies['num_ratings'] / (filtered_movies['num_ratings'] + m) * 
                                              filtered_movies['avg_rating'] + m / (filtered_movies['num_ratings'] + m) * C)
            filtered_movies = filtered_movies.sort_values('weighted_rating', ascending=False)
        
        # Display recommendations
        st.subheader("Recommended Popular Movies")
        if len(filtered_movies) > 0:
            display_cols = ['title', 'genres', 'avg_rating', 'num_ratings']
            if pop_sort == 'Weighted Rating':
                display_cols.append('weighted_rating')
            st.dataframe(filtered_movies[display_cols].head(20), use_container_width=True)
        else:
            st.warning(f"No movies found with the genre: {genre_filter}")
    
    # Tab 3: Content-based recommendations
    with tabs[2]:
        st.header("Content-Based Recommendation")
        st.markdown("""
        <p style='color: {}'>This approach recommends movies similar to a movie you specify, based on genre similarity.</p>
        """.format(COLORS['fg']), unsafe_allow_html=True)
        
        # Build content-based model when this tab is active
        with st.spinner("Building content-based model..."):
            cosine_sim, indices, movies_sample = build_content_based_model(movies_df, num_sample=5000)
        
        # Controls for content-based recommendations
        # Only offer movies that we know are in our sample
        available_titles = sorted(indices.index.tolist())
        
        # Add a search box to make it easier to find movies
        search_title = st.text_input("Search for a movie:", key="movie_search")
        
        if search_title:
            # Filter titles based on search
            filtered_titles = [title for title in available_titles if search_title.lower() in title.lower()]
            if filtered_titles:
                movie_title = st.selectbox("Select a movie:", filtered_titles, index=0, key='cb_movie')
            else:
                st.warning(f"No movies found matching '{search_title}'")
                movie_title = st.selectbox("Select a movie:", available_titles[:100], index=0, key='cb_movie')
        else:
            # Show a small subset of movies to prevent UI lag
            movie_title = st.selectbox("Select a movie:", available_titles[:100], index=0, key='cb_movie')
            
        num_recommendations = st.slider("Number of recommendations:", 5, 20, 10, key='cb_num')
        
        if st.button("Get Similar Movies", key='cb_button'):
            with st.spinner("Finding similar movies..."):
                recommendations = get_content_recommendations(movie_title, cosine_sim, indices, movies_sample, num_recommendations)
                
                # Display selected movie info
                try:
                    selected_movie = movies_sample[movies_sample['title'] == movie_title].iloc[0]
                    st.subheader(f"Selected Movie: {selected_movie['title']}")
                    st.markdown(f"**Genres:** {selected_movie['genres']}")
                except (IndexError, KeyError):
                    st.error(f"Movie '{movie_title}' not found in the sample dataset.")
                
                # Display recommendations
                st.subheader("Similar Movies")
                if recommendations is not None and not recommendations.empty:
                    # Format the similarity score as percentage
                    recommendations['similarity'] = (recommendations['similarity_score'] * 100).round(1).astype(str) + '%'
                    st.dataframe(recommendations[['title', 'genres', 'similarity']], use_container_width=True)
                else:
                    st.error("Sorry, couldn't find recommendations for this movie.")
    
    # Tab 4: Collaborative filtering recommendations
    with tabs[3]:
        st.header("Collaborative Filtering Recommendation")
        st.markdown("""
        <p style='color: {}'>This approach recommends movies based on similarities between users' preferences.</p>
        """.format(COLORS['fg']), unsafe_allow_html=True)
        
        # Check if we can use collaborative filtering (enough users)
        if ratings_sample['userId'].nunique() < 10:
            st.error("Not enough users in the sample for meaningful collaborative filtering. Please increase the sample size.")
        else:
            # Set up tabs with much more padding to ensure spacing is visible
            collab_tabs = st.tabs(["         Your Ratings         ", "         Existing User ID         "])
            
            with collab_tabs[0]:
                st.subheader("Rate Some Movies")
                st.markdown("""
                <p style='color: {}'>Rate at least 5-10 movies you've watched to get personalized recommendations.</p>
                """.format(COLORS['fg']), unsafe_allow_html=True)
                
                # Initialize the user's ratings in session state if not already done
                if 'user_ratings' not in st.session_state:
                    st.session_state.user_ratings = []
                
                # Add movies to rate
                col1, col2 = st.columns([3, 1])
                with col1:
                    # Show a dropdown with popular movies to rate
                    popular_to_rate = build_popularity_model(movies_df, ratings_sample, min_ratings=50).head(100)
                    movie_options = popular_to_rate['title'].tolist()
                    
                    # Add a search box to find movies
                    search_query = st.text_input("Search for a movie to rate:", key="rate_movie_search")
                    if search_query:
                        filtered_options = [m for m in movies_df['title'].tolist() if search_query.lower() in m.lower()]
                        # Limit to first 100 matches to avoid overwhelming the dropdown
                        movie_to_rate = st.selectbox("Select a movie:", filtered_options[:100], key='movie_to_rate')
                    else:
                        movie_to_rate = st.selectbox("Select a movie to rate:", movie_options, key='movie_to_rate')
                    
                    # Get the movieId for the selected movie
                    movie_id = movies_df[movies_df['title'] == movie_to_rate]['movieId'].values[0]
                
                with col2:
                    rating = st.slider("Your Rating:", min_value=0.5, max_value=5.0, value=3.0, step=0.5, key='new_rating')
                    
                    # Add button to add the rating
                    if st.button("Add Rating", key='add_rating'):
                        # Check if this movie is already rated
                        existing_idx = next((i for i, r in enumerate(st.session_state.user_ratings) if r['movieId'] == movie_id), None)
                        
                        if existing_idx is not None:
                            # Update existing rating
                            st.session_state.user_ratings[existing_idx]['rating'] = rating
                            st.success(f"Updated rating for '{movie_to_rate}'")
                        else:
                            # Add new rating
                            st.session_state.user_ratings.append({
                                'movieId': movie_id,
                                'title': movie_to_rate,
                                'rating': rating
                            })
                            st.success(f"Added rating for '{movie_to_rate}'")
                
                # Display current ratings
                if st.session_state.user_ratings:
                    st.subheader("Your Ratings")
                    ratings_df = pd.DataFrame(st.session_state.user_ratings)
                    st.dataframe(ratings_df[['title', 'rating']].sort_values('rating', ascending=False), use_container_width=True)
                    
                    # Add a button to clear all ratings
                    if st.button("Clear All Ratings"):
                        st.session_state.user_ratings = []
                        st.success("All ratings cleared!")
                    
                    # Controls for recommendations
                    cf_num = st.slider("Number of recommendations:", 5, 20, 10, key='user_cf_num')
                    
                    # Add button for recommendations
                    if st.button("Get Recommendations", key='user_cf_button'):
                        if len(st.session_state.user_ratings) < 3:
                            st.warning("Please rate at least 3 movies for better recommendations.")
                        
                        with st.spinner("Building a personalized model based on your ratings..."):
                            # Build the model with user input
                            svd = build_collaborative_filtering_model_with_user_input(ratings_sample, st.session_state.user_ratings)
                            
                            # Get recommendations
                            recommendations = get_user_based_recommendations(
                                st.session_state.user_ratings, 
                                svd, 
                                movies_df, 
                                ratings_sample,
                                cf_num
                            )
                            
                            # Display recommendations
                            st.subheader("Recommended Movies For You")
                            if recommendations is not None and not recommendations.empty:
                                st.dataframe(recommendations[['title', 'genres', 'predicted_rating']], use_container_width=True)
                            else:
                                st.error("Sorry, couldn't generate recommendations based on your ratings.")
                else:
                    st.info("You haven't rated any movies yet. Rate some movies to get recommendations!")
            
            with collab_tabs[1]:
                st.subheader("Use Existing User ID")
                st.markdown("""
                <p style='color: {}'>Use an existing user ID from the MovieLens dataset to see recommendations.</p>
                """.format(COLORS['fg']), unsafe_allow_html=True)
                
                # Build collaborative filtering model
                with st.spinner("Building collaborative filtering model..."):
                    svd = build_collaborative_filtering_model(ratings_sample)
                
                # Controls for collaborative filtering
                col1, col2 = st.columns(2)
                with col1:
                    user_id = st.number_input("Enter User ID:", min_value=1, max_value=int(ratings_sample['userId'].max()), step=1, key='cf_user')
                with col2:
                    cf_num = st.slider("Number of recommendations:", 5, 20, 10, key='cf_num')
                
                if st.button("Get Recommendations", key='cf_button'):
                    # Check if the user exists in our sample
                    if user_id in ratings_sample['userId'].values:
                        with st.spinner("Finding movies you might like..."):
                            user_movies = ratings_sample[ratings_sample['userId'] == user_id].merge(movies_df[['movieId', 'title']], on='movieId')
                            
                            # Display the user's top rated movies
                            st.subheader(f"User {user_id}'s Top Rated Movies")
                            top_user_movies = user_movies.sort_values('rating', ascending=False).head(5)
                            st.dataframe(top_user_movies[['title', 'rating']], use_container_width=True)
                            
                            # Get recommendations
                            recommendations = get_cf_recommendations(user_id, svd, movies_df, ratings_sample, cf_num)
                            
                            # Display recommendations
                            st.subheader("Recommended Movies")
                            if recommendations is not None and not recommendations.empty:
                                st.dataframe(recommendations[['title', 'genres', 'predicted_rating']], use_container_width=True)
                            else:
                                st.error("Sorry, couldn't generate recommendations for this user.")
                    else:
                        st.error(f"User {user_id} not found in the dataset sample. Try a different user ID or increase the sample size.")
    
    # Footer
    st.markdown("""
    <div style='margin-top: 50px; text-align: center; color: {}; font-size: small;'>
        Movie Recommendation System | Developed with Streamlit | Data from MovieLens
    </div>
    """.format(COLORS['bg_s']), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
