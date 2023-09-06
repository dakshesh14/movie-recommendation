import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

DATASET_PATH = 'datasets/'


class Model:
    def __init__(self):
        self.movies, self.credit = self.load_data()
        self.movies = self.preprocess_data()
        self.sig, self.indices = self.train()

    def load_data(self):
        movies = pd.read_csv(os.path.join(
            DATASET_PATH, 'tmdb_5000_movies.csv')
        )
        credit = pd.read_csv(os.path.join(
            DATASET_PATH, 'tmdb_5000_credits.csv')
        )
        return movies, credit

    def preprocess_data(self):

        movies, credit = self.load_data()

        movies = movies.merge(credit, on="title")

        movies = movies[['id', 'title', 'overview', 'genres', 'keywords',
                         'vote_average', 'vote_count', 'popularity', 'release_date', 'cast', 'runtime']]

        movies.columns = map(str.lower, movies.columns)

        movies['genres'] = movies['genres'].apply(eval)
        movies['genres'] = movies['genres'].apply(
            lambda x: [d['name'] for d in x])

        movies['keywords'] = movies['keywords'].apply(eval)
        movies['keywords'] = movies['keywords'].apply(
            lambda x: [d['name'] for d in x]
        )

        movies['release_date'] = pd.to_datetime(
            movies['release_date'], errors='coerce'
        )

        movies['cast'] = movies['cast'].apply(eval)
        movies['cast'] = movies['cast'].apply(lambda x: [d['name'] for d in x])

        movies.dropna(inplace=True)

        movies['overview'] = movies['overview'].apply(lambda x: x.split())

        # converting list to string
        movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x))
        movies['keywords'] = movies['keywords'].apply(lambda x: ' '.join(x))
        movies['cast'] = movies['cast'].apply(lambda x: ' '.join(x))
        movies['overview'] = movies['overview'].apply(lambda x: ' '.join(x))

        # final column for training
        movies['combined'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + \
            movies['title'] + movies['release_date'].apply(
                lambda x: str(x.year) + " " + str(x.month) + " " + str(x.day)
        )
        movies['title_lower'] = movies['title'].apply(lambda x: x.lower())

        C = movies['vote_average'].mean()
        m = movies['vote_count'].quantile(0.9)
        v = movies['vote_count']
        R = movies['vote_average']

        movies['weighted_average'] = (v/(v+m) * R) + (m/(m+v) * C)

        return movies

    def train(self):
        movies = self.movies
        tfv = TfidfVectorizer(max_features=5000, stop_words='english')
        tfv_matrix = tfv.fit_transform(movies['combined'])

        sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

        indices = pd.Series(
            movies.index, index=movies['title']).drop_duplicates()

        return sig, indices

    def recommend(self, title, x=11):
        movies = self.movies
        sig = self.sig

        index = movies[movies['title_lower'] == title.lower()].index[0]

        if index < 0:
            return {
                'detail': {},
                'recommendations': []
            }

        sig_scores = list(enumerate(sig[index]))
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
        sig_scores = sig_scores[1:11]
        movie_index = [i[0] for i in sig_scores]
        return {
            'detail': movies.iloc[index].to_dict(),
            'recommendations': movies[
                    ['title', 'vote_count', 'vote_average', 'release_date', 'genres',]
                ].iloc[movie_index].to_dict('records')
        }

    def get_top_x_movies(self, x=10):
        movies = self.movies
        return movies[['id', 'title', 'vote_count', 'vote_average', 'release_date', 'genres', 'weighted_average', 'overview', 'runtime',]].sort_values('weighted_average', ascending=False).head(x).to_dict('records')
