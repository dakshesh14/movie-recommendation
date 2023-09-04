# Movie Recommendation using Content Based Filtering

This is a movie recommendation AI as in API. It uses content based filtering to recommend movies. This API gives you top 'x' numbers of movies based on it's weighted average popularity which is calculated using Bayesian Average. The API also gives you the list of movies which are similar to the movie you have given as input. The similarity is calculated using sigmoid kernel.

## How to setup the API

This API is build using FastAPI. Following is the procedure to setup the API, assuming you have python3 installed in your system:

1. Clone the repository

```bash
git clone
```

2. Install the dependencies

```bash
pip install -r requirements.txt
```

3. Run the API

```bash
uvicorn main:app --reload
```

## How to use the API

The API has two endpoints:

1. /top-movies
2. /recommend-movies/{movie_id}

## Acknowledgement

- [FastAPI](https://fastapi.tiangolo.com/)
- [Pandas](https://pandas.pydata.org/)
- [TMDB](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [Scikit-learn](https://scikit-learn.org/stable/)
