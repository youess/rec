# coding: utf-8

'''
Here we use post filtering approach
'''

import numpy as np 
import pandas as pd 
import csv 


data = pd.read_csv("../data/ml-100k/u.data", sep="\t", header=None)
data.columns = ['UserID', 'ItemID', 'Rating', 'Timestamp']

# csv.register_dialect('movie_len', delimiter='|')
# path = "../data/ml-100k/u.item"
# # movies = pd.read_csv("../data/ml-100k/u.item", sep="|", header=None)
# with open(path) as f:
#     reader = csv.reader(f, movie_len)

path = "../data/ml-100k/u.item"
movies = pd.read_csv(path, sep="|", header=None, encoding='latin')

movies.columns = ["MovieId","MovieTitle","ReleaseDate","VideoReleaseDate","IMDbURL",
    "Unknown","Action","Adventure","Animation","Children","Comedy","Crime","Documentary",
    "Drama","Fantasy","FilmNoir","Horror","Musical","Mystery","Romance","SciFi",
    "Thriller","War","Western"]

movies.drop(["MovieTitle","ReleaseDate","VideoReleaseDate","IMDbURL"], axis=1, inplace=True)

# Model user item rating
ratings = pd.merge(data, movies, how='left', left_on="ItemID", right_on="MovieId")
ratings.drop('MovieId', axis=1, inplace=True)

ratings['nrate'] = (ratings['Rating'] > 3).astype(int)
scale_ratings = ratings.drop(['Rating', 'Timestamp'], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, roc_auc_score 


train, test = train_test_split(scale_ratings.iloc[:, 2:], test_size=0.2, random_state=42)
rf = RandomForestClassifier()
rf.fit(train.iloc[:, :-1], train['nrate'])

pred = rf.predict(test.iloc[:, :-1])
print("roc: ", roc_auc_score(pred, test['nrate']))

# let's choose one user
the_user = 943
# take out the user non rate movie feature
total_movies = movies['MovieId'].drop_duplicates()
rated_movie = data.loc[data['UserID'] == the_user, "ItemID"]

not_rate_movie = pd.DataFrame(
    total_movies[~total_movies.isin(rated_movie.values)].reset_index(drop=True))
#not_rate_movie['UserID'] = the_user 
#not_rate_movie['nrate'] = 0
not_rate_movie = not_rate_movie.merge(movies, on='MovieId', how='left')

rf_rec = rf.predict(not_rate_movie.iloc[:, 1:])
rf_rec = not_rate_movie.loc[rf_rec == 1]

# Context aware
rating_ctx = pd.merge(data, movies, how='left', left_on="ItemID", right_on="MovieId")
rating_ctx.drop('MovieId', axis=1, inplace=True)

# we could use timestamp as our context

# create context profile
ts = pd.to_datetime(rating_ctx['Timestamp'], unit='s', origin=pd.Timestamp('1960-10-01'))
rating_ctx['hours'] = ts.dt.hour

# for active user 943
UCP = rating_ctx[rating_ctx['UserID'] == 943].drop(['UserID', 'Rating', 'Timestamp'], axis=1)

UCP_pref = UCP.iloc[:, 1:].groupby('hours').sum()
# then normalize
UCP_pref_sc = (UCP_pref - UCP_pref.min(axis=0)) / (UCP_pref.max(axis=0) - UCP_pref.min(axis=0))

# generate context-aware rec
user_pred_by_ctx = pd.concat([rf_rec['MovieId'], rf_rec.iloc[:, 1:].dot(UCP_pref_sc.fillna(0).T)], axis=1) 
