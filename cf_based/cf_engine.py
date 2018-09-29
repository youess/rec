# coding: utf-8

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import mean_squared_error


np.random.seed(123)

data = pd.read_csv("../data/ml-100k/u.data", sep="\t", header=None)
data.columns = ['UserID', 'ItemID', 'Rating', 'Timestamp']

# do some data exploration

print("Rating count: \n", data['Rating'].value_counts())

# plt.hist(data['ItemID'].value_counts())
# plt.show()


print("User number: ", data['UserID'].nunique())
print("Item number: ", data['ItemID'].nunique())


# rating matrix
df = pd.pivot_table(data, index="UserID", columns="ItemID", 
    values=["Rating"], aggfunc=lambda x: x)
df.columns = df.columns.droplevel(0)

# calculate the sparsity
sparsity = df.notnull().sum().sum() / (df.shape[0] * df.shape[1])
print("User x Item matrix sparsity is : %.2f %%" % (100 * sparsity))

# convert into sparse matrix
df = np.array(df.fillna(0))

# split into train and test
rating_train, rating_test = train_test_split(df, test_size=0.33, random_state=42)

print("Training data shape: ", rating_train.shape)
print("Test data shape: ", rating_test.shape)

user_dist = 1 - cosine_distances(rating_train)

print("User similarity matrix shape: ", user_dist.shape)

# a bit bug that dense matrix dot sparse cost too much memory
# user_pred = user_dist.dot(rating_train.T.todense())
# Predicting the unknown rating value of item i for an active user u by calculating the
# weighted sum of all the users' ratings for the item.
user_pred = user_dist.dot(rating_train) / np.abs(user_dist).sum(axis=1, keepdims=True)

print("User predict matrix shape: ", user_pred.shape)


def get_mse(pred, actual):

    pred = pred[actual.nonzero()].flatten()
    actual = np.array(actual[actual.nonzero()]).flatten()
    return mean_squared_error(pred, actual)


print("Training mse: %.3f" % get_mse(user_pred, rating_train))
# So what's the fuck could directly use the previous user_pred
print("Test mse: %.3f" % get_mse(user_pred, rating_test))

# 上面我们用的是全部的相似用户，而实际上我们只需要考虑top-k个用户即可
k = 5
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(k, 'cosine')
neigh.fit(rating_train)

top_k_dist, top_k_users = neigh.kneighbors(rating_train, return_distance=True)

user_pred_k = np.zeros(rating_train.shape)
for i in range(rating_train.shape[0]):
    user_pred_k[i, :] = top_k_dist[i].reshape(-1, k).dot(rating_train[top_k_users[i]]) \
        / np.abs(top_k_dist[i]).sum(axis=0)
    print("process user: ", i)

print("Training mse: %.3f" % get_mse(user_pred_k, rating_train))
print("Test mse: %.3f" % get_mse(user_pred_k, rating_test))


# item based CF is similar as user based, just flip the shape of rating_train
