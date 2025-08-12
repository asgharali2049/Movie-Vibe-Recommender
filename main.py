import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os

# Page config
st.set_page_config(page_title="IMDb Movie Recommender", page_icon="🎬", layout="wide")

# Consistent, low-contrast CSS
st.markdown("""
    <style>
    .main {background-color: #ECEFF1;}
    .stButton>button {
        background-color: #5C6BC0;
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
    }
    .stSelectbox, .stMultiselect {
        background-color: #CFD8DC;
        border-radius: 8px;
        padding: 5px;
    }
    .stSelectbox>div>div>select, .stMultiselect>div>div>select {
        background-color: #CFD8DC;
        color: #212121;
        border-radius: 8px;
    }
    .movie-card {
        background-color: #CFD8DC;
        border: 1px solid #B0BEC5;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .movie-title {font-weight: bold; font-size: 1.2em; color: #574361;}
    .movie-text {color: #574361; font-size: 0.95em;}
    .sidebar .sidebar-content {background-color: #ECEFF1;}
    .sidebarupd-text {font-weight:500;color: #4C4268;font-size:1.em;}

    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    try:
        if not os.path.exists("imdb_top_1000.csv"):
            raise FileNotFoundError("imdb_top_1000.csv not found in the current directory.")
        movies = pd.read_csv("imdb_top_1000.csv")
        required_columns = ['Series_Title', 'Genre', 'Overview', 'Director', 'IMDB_Rating', 'Poster_Link']
        if not all(col in movies.columns for col in required_columns):
            raise KeyError(f"Dataset missing required columns: {required_columns}")

        movies = movies[required_columns].copy()
        movies = movies.rename(columns={
            'Series_Title': 'primaryTitle',
            'Genre': 'genres',
            'Overview': 'description',
            'Director': 'directors',
            'IMDB_Rating': 'averageRating',
            'Poster_Link': 'poster'
        })
        movies['primaryTitle'] = movies['primaryTitle'].str.lower().str.strip()
        movies = movies.drop_duplicates(subset='primaryTitle')
        movies['genres'] = movies['genres'].replace({'\\N': '', np.nan: ''}).str.replace(', ', ' | ')
        movies['description'] = movies['description'].replace({'\\N': '', np.nan: ''}).str.replace(r'[^\w\s]', '',
                                                                                                   regex=True).str.lower()
        movies['description'] = movies['description']
        movies['directors'] = movies['directors'].replace({'\\N': '', np.nan: ''})
        movies['averageRating'] = movies['averageRating'].fillna(0)
        movies['poster'] = movies['poster'].replace({'\\N': '', np.nan: ''})
        movies['combined_features'] = (movies['genres'] + ' ') + movies['description'] + ' ' + movies['directors']
        return movies.reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None


# Compute TF-IDF and cosine similarity
@st.cache_resource
def compute_tfidf_cosine(movies):
    try:
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return tfidf_matrix, cosine_sim
    except Exception as e:
        st.error(f"Error computing TF-IDF: {str(e)}")
        return None, None


# Recommendation function
def recommendations(filter_type, n, movie_title=None, preferred_genres=None, preferred_director=None, movies=None,
                    indices=None, cosine_sim=None):
    try:
        if filter_type == "Item-based":
            if not movie_title:
                return None, "Please select a movie for Item-based filtering."
            movie_title = movie_title.lower()
            if movie_title not in indices:
                return None, "Movie not found."
            idx = indices[movie_title]
            sim_score = list(enumerate(cosine_sim[idx]))
            input_movie = movies.iloc[idx]
            input_genres = set(input_movie['genres'].lower().split())
            input_director = input_movie['directors'].lower()
            for i, movie_idx in enumerate(sim_score):
                movie = movies.iloc[movie_idx[0]]
                weight = 1.0
                if movie['averageRating'] > 7.0:
                    weight += 0.5 * (movie['averageRating'] / 10.0)
                if movie['averageRating'] < 6.5:
                    weight *= 0.01
                movie_genres = set(movie['genres'].lower().split())
                if input_genres.intersection(movie_genres):
                    weight += 0.4
                if movie['directors'].lower() == input_director:
                    weight += 0.3
                sim_score[i] = (movie_idx[0], sim_score[i][1] * weight)
            sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)[1:n + 1]
            movie_indices = [i[0] for i in sim_score]
            return movies.iloc[movie_indices][
                ['primaryTitle', 'genres', 'description', 'directors', 'averageRating', 'poster']].to_dict(
                'records'), None



        elif filter_type == "Genre-based":
            if not preferred_genres:
                return None, "Please select at least one genre."
            filtered_movies = movies[
                movies['genres'].apply(lambda x: any(g.lower() in x.lower().split() for g in preferred_genres))]
            if filtered_movies.empty:
                return None, "No movies found for selected genres."
            filtered_movies = filtered_movies.sort_values(by='averageRating', ascending=False)
            return filtered_movies.head(n)[
                ['primaryTitle', 'genres', 'description', 'directors', 'averageRating', 'poster']].to_dict(
                'records'), None

        elif filter_type == "Director-based":
            if not preferred_director:
                return None, "Please select a director."
            filtered_movies = movies[movies['directors'].str.lower() == preferred_director.lower()]
            if filtered_movies.empty:
                return None, "No movies found for selected director."
            filtered_movies = filtered_movies.sort_values(by='averageRating', ascending=False)
            return filtered_movies.head(n)[
                ['primaryTitle', 'genres', 'description', 'directors', 'averageRating', 'poster']].to_dict(
                'records'), None
    except Exception as e:
        return None, f"Error in recommendations: {str(e)}"


# Load data
movies = load_data()
if movies is None:
    st.stop()

tfidf_matrix, cosine_sim = compute_tfidf_cosine(movies)
if tfidf_matrix is None:
    st.stop()

indices = pd.Series(movies.index, index=movies['primaryTitle'])

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        "<div class='sidebarupd-text'><em>IMDb Movie Recommender with item-based, genre-based, or director-based filtering using TF-IDF and IMDb Top 1000 dataset.",
        unsafe_allow_html=True)
    st.markdown("<div class='sidebarupd-text'><em>Author: Asghar Ali", unsafe_allow_html=True)
    st.markdown("[LinkedIn](https://www.linkedin.com/in/asghar-ali-47626b273/)", unsafe_allow_html=True)

# Main layout
st.title("IMDb Movie Recommender🎬")
st.markdown("<div class='movie-text'>Select a movie and filter type for personalized recommendations.</div>",
            unsafe_allow_html=True)

# Filter type
st.subheader("How do you want to filter movies?")
filter_type = st.selectbox("Filter by:", ["Item-based", "Genre-based", "Director-based"], key="filter_type")

# Conditional inputs
movie_input = None
preferred_genres = None
preferred_director = None
num_recommendations = st.slider("Recommendations:", 1, 10, 5, key="num_recs")

if filter_type == "Item-based":
    movie_input = st.selectbox("Choose a movie:", [""] + sorted(movies['primaryTitle'].tolist()), key="movie_input")
elif filter_type == "Genre-based":
    all_genres = sorted(set(' '.join(movies['genres']).split()))
    preferred_genres = st.multiselect("Preferred genres:", all_genres, key="genres_input")
elif filter_type == "Director-based":
    all_directors = sorted(set(movies['directors'].dropna()))
    preferred_director = st.selectbox("Select director:", [""] + all_directors, key="director_input")

# Get recommendations
if st.button("Get Recommendations", key="get_recs"):
    with st.spinner("Finding movies..."):
        recommended_movies, error = recommendations(filter_type, num_recommendations, movie_input, preferred_genres,
                                                    preferred_director, movies, indices, cosine_sim)

    if error:
        st.error(error)
    else:
        header = f"Recommendations ({filter_type})"
        if filter_type == "Item-based":
            header = f"Recommendations for '{movie_input.title()}' (Item-based)"
        elif filter_type == "Genre-based":
            header = f"Recommendations for genres: {', '.join(preferred_genres)}"
        elif filter_type == "Director-based":
            header = f"Recommendations for director: {preferred_director}"
        st.subheader(header)

        for movie in recommended_movies:
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(movie['poster'] if movie.get('poster') else "https://via.placeholder.com/150", width=150)
                with col2:
                    st.markdown(f"""
                        <div class='movie-card'>
                            <span class='movie-title'>{movie['primaryTitle'].title()}</span><br>
                            <span class='movie-text'>
                                <b>Genres:</b> {movie['genres']}<br>
                                <b>Director:</b> {movie['directors']}<br>
                                <b>IMDB Rating:</b> {movie['averageRating']}/10<br>
                                <b>Plot:</b> {movie['description'][:200]}...
                            </span>
                        </div>
                    """, unsafe_allow_html=True)

        # Genre distribution chart
        genre_counts = {}
        for movie in recommended_movies:
            for genre in movie['genres'].split(' | '):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        labels = list(genre_counts.keys())
        data = list(genre_counts.values())

        chart_html = f"""
            <div style="width: 100%; height: 350px;">
                <canvas id="genreChart"></canvas>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <script>
                    new Chart(document.getElementById('genreChart'), {{
                        type: 'pie',
                        data: {{
                            labels: {labels},
                            datasets: [{{
                                data: {data},
                                backgroundColor: ["#006d77", "#83c5be", "#edf6f9", "#ffddd2", "#e29578"]
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            plugins: {{
                                legend: {{ position: 'top' }},
                                title: {{ display: true, text: 'Genre Distribution' }}
                            }}
                        }}
                    }});
                </script>
            </div>
        """
        st.components.v1.html(chart_html, height=400)

st.subheader("Rate and Review Our App")

# 1. User details
name = st.text_input("Your Good Name")
email = st.text_input("Your Email")
# 3. Feedback message
rating = st.slider('Rate your Experience: ', min_value=1, max_value=5, step=1)
feedback = st.text_area("Tell us what you think!")

# 4. Submit Button
if st.button("Submit Feedback", key="submit_feedback"):
    if not name.strip() or not email.strip() or not feedback.strip():
        st.warning("Please fill in all fields.")
    else:
        # Prepare feedback data
        new_feedback = pd.DataFrame([[datetime.now().strftime('%Y-%m-%d %H:%M:%S'), name, email, rating, feedback]],
                                    columns=["timestamp", "name", "email", "rating", "feedback"])

        # Save to CSV
        file_path = "feedback.csv"
        if os.path.exists(file_path):
            new_feedback.to_csv(file_path, mode='a', header=False, index=False)
        else:
            new_feedback.to_csv(file_path, index=False)

        st.success("🎉 Thanks for your feedback!")

# 5. Display average rating if feedback file exists
if os.path.exists("feedback.csv"):
    df = pd.read_csv("feedback.csv")
    if not df.empty and "rating" in df.columns:
        avg_rating = round(df["rating"].mean(), 2)
        st.subheader(f"⭐ Average User Rating: **{avg_rating}/5**")
