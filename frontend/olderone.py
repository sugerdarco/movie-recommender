import streamlit as st
import requests
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import initialize_recommender, query_recommendations, fetch_all_titles

# AI graph done
rec = initialize_recommender()

def get_movie_poster(movie_id, api_key="f3f98776fabedfa7e3f1a8dddd61ace4"):

    base_url = "https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US".format(movie_id, api_key)
    response = requests.get(base_url)

    if response.status_code == 200:
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
        else:
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


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
        poster_url = get_movie_poster(movie_id)
        print(f"movie post: {poster_url}")
        movie["poster"] = poster_url  # directly add poster field

    # Now rec_movies contains poster in each dictionary
    print(f"rec_movies with poster: {rec_movies}")

    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]

    for i, movie in enumerate(rec_movies[:5]):
        with cols[i]:
            st.text(movie["title"])  # always show title
            if movie["poster"] is not None:  # show poster only if available
                # st.image(movie["poster"])
                img_html = f'<img src="{movie["poster"]}" loading="lazy" style="width:100%;">'
                st.markdown(img_html, unsafe_allow_html=True)
