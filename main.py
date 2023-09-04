from fastapi import FastAPI
from model.model import Model


model = Model()
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/top-movies")
async def top_movies():
    return model.get_top_x_movies(20)


@app.get("/recommend-movies/{movie_title}")
async def recommend_movies(movie_title: str):
    return model.recommend(movie_title)
