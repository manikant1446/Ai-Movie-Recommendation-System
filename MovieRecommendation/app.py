import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------
# 1. DATA PREPARATION (Data Filtering)
# --------------------------------------------------------
# Load the Kaggle movies dataset from the local CSV file
csv_path = os.path.join(os.path.dirname(__file__), 'movies.csv')
movies_df = pd.read_csv(csv_path)

# Prepare genres (replace commas with space) and create genre/description tags
movies_df['genre'] = movies_df['genres'].str.replace(',', ' ')
movies_df['tags'] = movies_df['genre'] + " " + movies_df['description']

# --------------------------------------------------------
# 2. SIMILARITY MATCHING 
# --------------------------------------------------------
# Convert the text data into numerical vectors
vectorizer = TfidfVectorizer(stop_words='english')
feature_vectors = vectorizer.fit_transform(movies_df['tags']).toarray()

# Calculate the cosine similarity between all movies
similarity_matrix = cosine_similarity(feature_vectors)

def get_recommendations(movie_title, top_n=20):
    """Finds movies similar to the given title."""
    # Find the index of the selected movie
    movie_index = movies_df[movies_df['title'] == movie_title].index[0]
    
    # Get similarity scores for this movie compared to all others
    distances = similarity_matrix[movie_index]
    
    # Sort the movies based on similarity scores (highest first)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
    
    # Get the top N similar movies (skipping the first one, which is the movie itself)
    recommended_titles = []
    for i in movies_list[1:top_n+1]:
        recommended_titles.append(movies_df.iloc[i[0]].title)
        
    return recommended_titles

# --------------------------------------------------------
# 3. USER INTERFACE (Streamlit)
# --------------------------------------------------------
st.title("🎞️ AI Movie Recommendation System")

# Helper function to find matching movie title
def find_movie(query):
    if not query or not query.strip():
        return None
    # Exact case-insensitive match first
    exact_match = movies_df[movies_df['title'].str.lower() == query.strip().lower()]
    if not exact_match.empty:
        return exact_match.iloc[0]['title']
    # Substring match next
    partial_match = movies_df[movies_df['title'].str.contains(query.strip(), case=False, na=False)]
    if not partial_match.empty:
        return partial_match.iloc[0]['title']
    return None

# All available genres extracted from the dataset
all_genres = sorted(set(
    genre.strip()
    for genres in movies_df['genre']
    for genre in genres.split()
))

# Columns side-by-side: Adjusted ratio so the search button box is smaller and fits the 🔍 icon perfectly
col_search, col_btn, col_filter = st.columns([15, 1.7, 5])

with col_search:
    search_query = st.text_input(
        label="",
        placeholder="Search for a movie...",
        label_visibility="collapsed"
    )

with col_btn:
    search_clicked = st.button("🔍", use_container_width=True)

with col_filter:
    selected_genre = st.selectbox(
        label="",
        options=["All Categories"] + all_genres,
        label_visibility="collapsed"
    )

# Perform recommendations if search is active
if search_query or search_clicked:
    if search_query.strip():
        matched_movie = find_movie(search_query)
        
        if matched_movie:
            # Get details of the matched movie
            movie_row = movies_df[movies_df['title'] == matched_movie].iloc[0]
            
            # 1. Show searched movie details first in a rounded card
            st.subheader("Search Result :")
            with st.container(border=True):
                st.markdown(f"### **{movie_row['title']}** ({int(movie_row['year'])})")
                st.write(f"⭐ **Rating:** {movie_row['rating']}/10")
                st.write(f"🎭 **Genres:** {movie_row['genres']}")
                st.write(f"🎬 **Director:** {movie_row['director']}")
                st.write(f"👥 **Cast:** {movie_row['cast']}")
                st.write(f"📝 **Description:** {movie_row['description']}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 2. Show recommended movies below it
            with st.spinner('Analyzing genres and descriptions...'):
                recommendations = get_recommendations(matched_movie, top_n=50)
                
                st.subheader("🎬 Recommended Movies:")
                
                # Filter recommendations: exclude searched movie, avoid duplicates, and filter by category (max 6)
                filtered_recs = []
                for movie in recommendations:
                    # Skip if it is the searched movie itself
                    if movie.strip().lower() == matched_movie.strip().lower():
                        continue
                    # Skip if already added
                    if movie in filtered_recs:
                        continue
                    # Filter by genre
                    if selected_genre == "All Categories" or selected_genre in movies_df[movies_df['title'] == movie]['genre'].values[0]:
                        filtered_recs.append(movie)
                    if len(filtered_recs) == 6:
                        break
                
                if filtered_recs:
                    # Display recommendations in a grid of 3 rounded cards per row
                    cols_per_row = 3
                    for i in range(0, len(filtered_recs), cols_per_row):
                        row_recs = filtered_recs[i:i+cols_per_row]
                        rec_cols = st.columns(len(row_recs))
                        for idx, movie in enumerate(row_recs):
                            rec_row = movies_df[movies_df['title'] == movie].iloc[0]
                            with rec_cols[idx]:
                                with st.container(border=True):
                                    st.markdown(f"### **{rec_row['title']}** ({int(rec_row['year'])})")
                                    st.write(f"⭐ **Rating:** {rec_row['rating']}/10")
                                    st.caption(f"**Genres:** {rec_row['genres']}")
                                    st.write(rec_row['description'])
                else:
                    st.info("No recommendations matching the selected genre. Try switching the category filter.")
        else:
            st.error(f"Could not find any movie matching '{search_query}'. Try searching for 'The Matrix', 'Inception', or 'Titanic'.")
    else:
        st.warning("Please type a movie name to search.")