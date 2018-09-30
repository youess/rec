# coding: utf-8

import numpy as np 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_distances


path = "../data/anonymous-msweb.test.txt"
raw_data = pd.read_csv(path, header=None, skiprows=7)
raw_data.shape

# col 1: A for case id, V for the user, C for case id that the user has accessed
raw_data.iloc[:, 0].value_counts()

# col2: id to represent users and items
# col3: the description of website area
# col4: URL

## User activity format, which user clicked the web
user_activity = raw_data.loc[raw_data[0] != 'A', :]
user_activity = user_activity.loc[:, :1]
user_activity.columns = ['category','value']
user_activity.reset_index(drop=True, inplace=True)

# unique number of webid
user_activity.loc[user_activity['category'] == 'V', 'value'].nunique()

# unique number of userid
user_activity.loc[user_activity['category'] == 'C', 'value'].nunique()

# create user item rating matrix
tmp = 0
nextrow = False 
lastindex = user_activity.index[-1]
user_activity["userid"] = 0
user_activity["webid"] = 0

for index, row in user_activity.iterrows():

    if index <= lastindex:
        if row['category'] == 'C':
            tmp = 0
            userid = row['value']
            user_activity.loc[index, 'userid'] = userid 
            user_activity.loc[index, 'webid'] = userid 
            tmp = userid 
            nextrow = True 
        elif nextrow and row['category'] == 'V':
            webid = row['value']
            user_activity.loc[index, 'userid'] = tmp 
            user_activity.loc[index, 'webid'] = webid
            if index != lastindex and user_activity.loc[index+1, 'category'] == 'C':
                nextrow = False 
                tmp = 0 

user_activity.head(10)

# remove unwanted row
user_activity = user_activity[user_activity['category'] == "V" ]
user_activity = user_activity[['userid','webid']]
user_activity_sort = user_activity.sort_values('webid', ascending=True)
user_activity_sort['rating'] = 1
ratmat = user_activity_sort.pivot(index='userid', columns='webid', values='rating').fillna(0)
print(ratmat.shape)

## Item profiling
items = raw_data.loc[raw_data[0] == "A"]
items.columns = ['record','webid','vote','desc','url']
items = items[['webid','desc']]

print("item nunique: ", items['webid'].nunique())

items2 = items[items['webid'].isin(user_activity['webid'].tolist())]
items_sort = items2.sort_values('webid', ascending=True)

from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(stop_words ="english",max_features = 100,ngram_range=
(0,3),sublinear_tf =True)
x = v.fit_transform(items_sort['desc'])
itemprof = x.todense()

# User profiling 
from scipy import linalg, dot
userprof = dot(ratmat,itemprof)/linalg.norm(ratmat)/linalg.norm(itemprof)

# similarity
import sklearn.metrics
similarityCalc = sklearn.metrics.pairwise.cosine_similarity(userprof, itemprof, dense_output=True)

final_pred= np.where(similarityCalc>0.6, 1, 0)

indexes_of_user = np.where(final_pred[213] == 1)

