import pandas as pd
import numpy as np
import requests
import re

# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Fetch the data from the URL
response = requests.get(myurl)

# Split the data into lines and then split each line using "::"
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

# Create a DataFrame from the movie data
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)

# Load my data
url = "https://raw.githubusercontent.com/t5961p/stat542proj4/main/top10_genres.csv"
myData = pd.read_csv(url)
# S matrix
url = "https://raw.githubusercontent.com/t5961p/stat542proj4/main/S_matrix.csv"
S_df = pd.read_csv(url)


def get_displayed_movies():
    return movies[1570:1710]#movies.head(100)

## get recommended movies based on user ratings (System II)
def get_recommended_movies(new_user_ratings):
    ## NOTE: new_user_ratings is dict {movie_id: rating}. Need to convert to numpy array
    ratings_array = np.empty((3706))
    ratings_array.fill(np.nan)
    
    for key, value in new_user_ratings.items():
        string_id = 'm' + str(key) # find index in S matrix
        index = S_df.columns.get_loc(string_id)
        ratings_array[index] = value

    # Call myIBCF function
    top10_id = myIBCF(ratings_array.reshape((1, -1)))[0]
    
    df = pd.DataFrame()

    for i in range(10):
        string_id = top10_id[i]
        num_id = re.sub(r"\D", "", string_id) # remove non numeric character
        df = pd.concat([df, movies.loc[movies['movie_id']==int(num_id)] ], ignore_index=True)

    return df

## get popular movies based on genres (System I)
def get_popular_movies(genre: str):

    ## get movies id with genre
    id_genre = myData[genre].values

    df = pd.DataFrame()

    for i in range(10):
        df = pd.concat([df, movies.loc[movies['movie_id']==id_genre[i]] ], ignore_index=True)

    return df

def myIBCF(newuser):
    """
    my IBCF function.
    Input:
        newuser: (3706-by-1 array) ratings for the 3706 movies from a new user.

    Output:
        top10_id: (1d array) top 10 movies id
        top10_ratings: (1d array) top 10 movies ratings
    """

    S_matrix = S_df.values
    # S_matrix = np.nan_to_num(S_matrix) # fill na values with 0

    ## extract the movies entries that were rated 
    newuser = newuser.reshape(-1)
    mask_u = ~np.isnan(newuser)
    
    # w = newuser[mask_u]
    pred = np.empty_like(newuser)
    
    ## predict unrated movies
    for i in range(S_matrix.shape[0]):
        if np.isnan(newuser[i]):
            # choose indices that the new user has rated and are in 30-nn of movie i 
            mask = (~np.isnan(S_matrix[i,:])) & mask_u
             
            w = newuser[mask]
            # denom = np.sum(S_matrix[i, mask])
            if w.shape[0] > 0:
                pred[i] = np.dot(S_matrix[i, mask], w) / np.sum(S_matrix[i, mask])
            else:
                pred[i] = np.nan
        else:
            pred[i] = np.nan # no need to predict rated movies

    ## extract columns names of ten recommendations
    top10_indices = np.argsort(-pred)
    movie_id = S_df.columns.values

    top10_id = np.empty(10, dtype=object)
    top10_ratings = np.empty(10)
    for i in range(10):
        top10_id[i] = movie_id[top10_indices[i]]
        top10_ratings[i] = pred[top10_indices[i]]


    return top10_id, top10_ratings