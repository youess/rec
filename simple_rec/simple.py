# coding: utf-8

import pandas as pd 


# Load the data
data = pd.read_csv('../data/movie_rating.csv')

print(data.info())

df = pd.pivot_table(data, index="title", columns="critic", 
    values=["rating"], aggfunc=lambda x: x)
df.columns = df.columns.droplevel(0)
print(df)


# Calculate the similarity between users
user_cor = df.corr(method="pearson")


# Predict the unknown rating for users
## case user Toby

# step 1. find the list of movie that user not rate
the_user = "Toby"
movie_not_rate_by_user = df.index[df[the_user].isnull()].tolist()

# step 2. isolate non-rate movie list, so we get other user ratings
rating_of_other_user = data[data['title'].isin(movie_not_rate_by_user)]
# add similarity between toby and other user
sim_user = user_cor[the_user].reset_index().rename(columns={the_user: "similarity"})
# sim_user = sim_user[sim_user['critic'] != the_user].reset_index(drop=True)
rating_of_other_user = rating_of_other_user.merge(sim_user, how='left', on='critic')

# step 3. multiply the rating and similarity
rating_of_other_user['sim_rating'] = rating_of_other_user['rating'] * rating_of_other_user['similarity']

# step 4. sum up the sim_rating and divide by toby similar user value
sim_rating = rating_of_other_user.groupby('title')['similarity', 'sim_rating'].sum()
sim_rating = sim_rating['sim_rating'] / sim_rating['similarity']

# generate recommendate movie by some metric
## eg: rate score > mean of Toby rating
mean_rating = df[the_user].mean()  # remove nan
rec_list = sim_rating.index[sim_rating > mean_rating].tolist()

print(rec_list)


def generate_rec(the_user):
    movie_not_rate_by_user = df.index[df[the_user].isnull()].tolist()

    # step 2. isolate non-rate movie list, so we get other user ratings
    rating_of_other_user = data[data['title'].isin(movie_not_rate_by_user)]
    # add similarity between toby and other user
    sim_user = user_cor[the_user].reset_index().rename(columns={the_user: "similarity"})
    # sim_user = sim_user[sim_user['critic'] != the_user].reset_index(drop=True)
    rating_of_other_user = rating_of_other_user.merge(sim_user, how='left', on='critic')
    
    # step 3. multiply the rating and similarity
    rating_of_other_user['sim_rating'] = rating_of_other_user['rating'] * rating_of_other_user['similarity']
    
    # step 4. sum up the sim_rating and divide by toby similar user value
    sim_rating = rating_of_other_user.groupby('title')['similarity', 'sim_rating'].sum()
    sim_rating = sim_rating['sim_rating'] / sim_rating['similarity']
    
    # generate recommendate movie by some metric
    ## eg: rate score > mean of Toby rating
    mean_rating = df[the_user].mean()  # remove nan
    rec_list = sim_rating.index[sim_rating > mean_rating].tolist()

    return rec_list, list(zip(sim_rating.index.tolist(), sim_rating.round(2).tolist())) 

# other users
for user in df.columns.tolist():
    rec_list, raw_rec = generate_rec(user)
    print("rec user [ %s ] movie list: %s and raw rec result: %s" % (user, rec_list, raw_rec))

