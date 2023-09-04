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

        movies.dropna(inplace=True)

        movies = movies[['id', 'title', 'overview', 'genres', 'keywords',
                         'vote_average', 'vote_count', 'popularity', 'release_date', 'cast', 'runtime']]

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

        movies['overview'] = movies['overview'].apply(lambda x: x.split())

        movies['title_lower'] = movies['title'].apply(lambda x: x.lower())

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

        C = movies['vote_average'].mean()
        m = movies['vote_count'].quantile(0.9)
        v = movies['vote_count']
        R = movies['vote_average']

        movies['weighted_average'] = (v/(v+m) * R) + (m/(m+v) * C)

        return movies

    def train(self):
        movies = self.preprocess_data()
        tfv = TfidfVectorizer(min_df=3, max_features=None,
                              strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                              ngram_range=(1, 3), stop_words='english')
        tfv_matrix = tfv.fit_transform(movies['combined'])

        sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

        indices = pd.Series(
            movies.index, index=movies['title']).drop_duplicates()

        return sig, indices

    def recommend(self, title, x=11):
        movies = self.movies
        sig = self.sig

        index = movies[movies['title_lower'] == title.lower()]

        if len(index) == 0:
            return []

        index = index.index[0]
        sig_scores = list(enumerate(sig[index]))
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
        sig_scores = sig_scores[1:x]
        movie_index = [i[0] for i in sig_scores]
        return movies[
            ['id', 'title', 'vote_count', 'vote_average',
                'release_date', 'genres', 'overview', 'runtime',
             ]
        ].iloc[movie_index].to_dict('records')

    def get_top_x_movies(self, x=10):
        movies = self.movies
        return movies[['id', 'title', 'vote_count', 'vote_average', 'release_date', 'genres', 'weighted_average', 'overview', 'runtime',]].sort_values('weighted_average', ascending=False).head(x).to_dict('records')
