import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🎬",
    layout="wide"
)

# ================= LOAD FILE =================
df = pickle.load(open("netflix_df.pkl", "rb"))

# ================= BUILD MODEL =================
@st.cache_data
def build_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Tags'])  # change column if needed
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

cosine_sim = build_model(df)

# ================= CSS =================
st.markdown("""
<style>
body {
    background-color: #141414;
}

.stApp {
    background-color: #141414;
    color: white;
    font-family: 'Arial', sans-serif;
}

/* HEADER */
.netflix-header {
    font-size: 50px;
    font-weight: 900;
    color: #E50914;
    padding-top: 10px;
}

.subtitle {
    font-size: 20px;
    margin-bottom: 30px;
}

/* SEARCH LABEL */
label {
    color: white !important;
    font-size: 22px !important;
    font-weight: bold;
}

/* SEARCH INPUT BOX */
input {
    font-size: 20px !important;
    background-color: #000 !important;
    color: white !important;
}

/* MOVIE CARD */
.movie-card {
    background: #1f1f1f;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    transition: transform 0.3s ease;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.6);
}

.movie-card:hover {
    transform: scale(1.05);
    background: #262626;
}

.movie-title {
    font-size: 18px;
    font-weight: bold;
    color: white;
}

/* BUTTON */
.button-style button {
    background-color: black !important;
    color: white !important;
    font-size: 18px !important;
    border-radius: 8px !important;
    height: 50px !important;
    border: 2px solid #E50914 !important;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<div class='netflix-header'>NETFLIX</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>What do you want to watch today?</div>", unsafe_allow_html=True)

# ================= SEARCH =================
movie_name = st.text_input("Search for a movie or TV show")

# ================= RECOMMEND FUNCTION =================
def recommend(movie_name):
    movie_name = movie_name.lower()
    df['Title_lower'] = df['Title'].str.lower()

    matches = df[df['Title_lower'].str.contains(movie_name)]

    if matches.empty:
        return []

    idx = matches.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:7]
    movie_indices = [i[0] for i in sim_scores]

    return df['Title'].iloc[movie_indices].tolist()

# ================= BUTTON =================
st.markdown("<div class='button-style'>", unsafe_allow_html=True)
clicked = st.button("🍿 Recommend")
st.markdown("</div>", unsafe_allow_html=True)

# ================= OUTPUT =================
if clicked:
    if movie_name.strip() == "":
        st.warning("Please enter a movie name")
    else:
        recommendations = recommend(movie_name)

        if len(recommendations) == 0:
            st.error("Movie not found in database")
        else:
            st.subheader("Recommended for you")
            cols = st.columns(5)

            for i, rec in enumerate(recommendations):
                with cols[i % 5]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">🎬 {rec}</div>
                    </div>
                    """, unsafe_allow_html=True)
