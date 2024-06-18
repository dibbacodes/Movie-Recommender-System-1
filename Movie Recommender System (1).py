#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd


# In[38]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[39]:


movies.head(1)


# In[40]:


credits.head(1)


# In[41]:


movies = movies.merge(credits, on = 'title')


# In[42]:


movies.info()


# In[43]:


# genere, id, keywords, title, overview, cast, crew

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']] 


# In[44]:


movies


# In[45]:


movies.isnull().sum()


# In[46]:


movies.dropna(inplace=True)


# In[47]:


movies.isnull().sum()


# In[48]:


movies.duplicated().sum()


# In[49]:


movies.iloc[0].genres


# In[12]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# to ['Action', 'Adventure'.....]


# In[50]:


def convert (obj):
    L = []
    for i in obj:
        L.append(i['name'])
    return L


# In[51]:


convert(movies.iloc[0].genres)


# In[54]:


import ast
#using literal eval funtion to convert string to list
ast.literal_eval(movies.iloc[0].genres)


# In[55]:


def convert (obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[57]:


movies['genres'] = movies['genres'].apply(convert)


# In[58]:


movies


# In[59]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[60]:


movies.head()


# In[61]:


#extracting top 3 actor names
def convert2 (obj):
    L = []
    cnt=0;
    for i in ast.literal_eval(obj):
        if(cnt==3):
            break;
        else:
            L.append(i['name'])
            cnt+=1
    return L


# In[62]:


movies['cast'] = movies['cast'].apply(convert2)


# In[22]:


movies.head()


# In[63]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break;
    return L


# In[64]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[65]:


movies.head()


# In[66]:


# overview is still a string we want to convert it to a list as well
movies['overview'][0]


# In[67]:


movies['overview'] = movies['overview'].apply(lambda x: x.split())


# In[81]:


movies.head()


# In[28]:


# We now want to remove spaces between names to avoid confusion in 
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[29]:


movies.head()


# In[68]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']  


# In[69]:


movies.head()


# In[70]:


new_df = movies[['movie_id', 'title', 'tags']]


# In[32]:


new_df.head()


# In[71]:


# converting the list: tags to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# In[34]:


new_df.head()


# In[72]:


new_df['tags'][0]


# In[92]:


new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# In[73]:


new_df.head() 


# Vectorization

# In[74]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words = 'english')


# In[75]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[76]:


vectors


# In[77]:


cv.get_feature_names_out()


# In[78]:


# do steming for similar words
#[ love, loved, loving] = [love, love, love]

import nltk


# In[79]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[80]:


def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)


# In[81]:


#examples of stemming
ps.stem('dancing')


# In[82]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[83]:


new_df['tags']


# In[84]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[85]:


cv.get_feature_names_out()


# In[86]:


#cosine distance
from sklearn.metrics.pairwise import cosine_similarity


# In[87]:


#4806x4806 matrix with each row suggesting the similarity of a movie with others
similarity = cosine_similarity(vectors)


# In[89]:


similarity[0]
# we want to pair the similarity with index then sort
list(enumerate(similarity[0])) # this tells 0 index movie ka har movie index sath similarity


# In[91]:


#sort on basis of similarity
sorted(list(enumerate(similarity[0])), reverse = True, key = lambda x:x[1])


# In[92]:


def recommend (movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[99]:


recommend('Tangled')


# In[ ]:




