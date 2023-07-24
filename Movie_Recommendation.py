import numpy as np
import pandas as pd

credits = pd.read_csv("credits.csv")
movie = pd.read_csv("movies.csv")

movies = movie.merge(credits, on = "title")


movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

movies.dropna(inplace=True)

import ast

def convert(obj):
    L= []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies["genres"] = movies["genres"].apply(convert).copy()
movies["keywords"] = movies["keywords"].apply(convert).copy()

def convert3(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i["name"])
            counter += 1
        else:
            break
    return L
movies["cast"] = movies["cast"].apply(convert3).copy()

def convert4(obj):
    L= []
    for i in ast.literal_eval(obj):
        if i["job"]== "Director":
            L.append(i["name"])
    return L

movies['crew'] = movies['crew'].apply(convert4).copy()

movies["overview"] = movies["overview"].apply(lambda x:x.split()).copy()

movies["genres"] = movies["genres"].apply(lambda x:[i.replace(" ", "") for i in x]).copy()
movies["keywords"] = movies["keywords"].apply(lambda x:[i.replace(" ", "") for i in x]).copy()
movies["cast"] = movies["cast"].apply(lambda x:[i.replace(" ", "") for i in x]).copy()
movies["crew"] = movies["crew"].apply(lambda x:[i.replace(" ", "") for i in x]).copy()

movies["tags"] = movies["genres"]+movies["keywords"]+movies["cast"]+movies["crew"].copy()

new_movie = movies[["movie_id", "title", "tags"]].copy()

new_movie['tags'] = new_movie['tags'].apply(lambda x: ' '.join(x)).copy()
new_movie['tags'] = new_movie['tags'].apply(lambda x:x.lower()).copy()

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
victors = cv.fit_transform(new_movie['tags']).toarray()

import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_movie["tags"] = new_movie["tags"].apply(stem).copy()

from sklearn.metrics.pairwise import cosine_similarity

simrity = cosine_similarity(victors)

#sorted(list(enumerate(simrity[0])), reverse= True, key=lambda x:x[1])[1:6]

def recomend(movie):
    movie_index = new_movie[new_movie['title'] == movie].index[0]
    distance = simrity[movie_index]
    movie_list = sorted(list(enumerate(distance)), reverse= True, key=lambda x:x[1])[1:6]

    for i in movie_list:
        print(new_movie.iloc[i[0]].title)

print(recomend("Divergent"))
