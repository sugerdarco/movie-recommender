import streamlit as st
import requests
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import initialize_recommender, query_recommendations, fetch_all_titles, fetch_movies_by_ids

# AI graph done
rec = initialize_recommender()

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


st.header('Movie Recommender System')

movies = fetch_all_titles()
# movies format: [{title: "", movie_id: ""}, {}, ...]
movie_list = [m["title"] for m in movies]

# Select movie from dropdown
selected_title = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)
# Map selected title back to movie_id
selected_movie_id = next((m["movie_id"] for m in movies if m["title"] == selected_title), None)


if st.button('Show Recommendation'):
    rec_movies = query_recommendations(rec, selected_movie_id) # return type = [{title, movie_id, score}]
    print(f"rec_movie: {rec_movies}")
    # rec_movie_data
    for movie in rec_movies:
        movie_id = movie["movie_id"]
        poster_url = fetch_poster(movie_id)
        print(f"movie post: {poster_url}")
        movie["poster"] = poster_url  # directly add poster field

    # Now rec_movies contains poster in each dictionary
    print(f"rec_movies with poster: {rec_movies}")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(rec_movies[0]["title"]) # title
        st.image(rec_movies[0]["poster"]) # poster
    with col2:
        st.text(rec_movies[1]["title"])
        st.image(rec_movies[1]["poster"])

    with col3:
        st.text(rec_movies[2]["title"])
        st.image(rec_movies[2]["poster"])
    with col4:
        st.text(rec_movies[3]["title"])
        st.image(rec_movies[3]["poster"])
    with col5:
        st.text(rec_movies[4]["title"])
        st.image(rec_movies[4]["poster"])
