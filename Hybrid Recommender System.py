#!/usr/bin/env python
# coding: utf-8

# ## Recommender System 
# 
# The marketing analytics team is currently working on building and deploying a hybrid recommender system that will suggest items to customers based on their previous purchase history. This will help a company offer and suggest specialized products to its customers. Recommender systems are used by Amazon, Netflix, Facebook, LinkedIn, and many more large corporations allowing them to analyze large amounts of user data and quickly offer customized recommendations.
# 
# Two common recommender systems are collaborative filtering and content-based. 
# 
# #### Collaborative Filtering
# 
# A collaborative filtering system suggests items to users based on the similarity of preferences and choices based on other users. For example, if user 1 bought items A, B, C, and D and user 2 bought items A, B, and C the system would recommend user 2 item D. The key assumption is that there are common similarities between users and that similar users in the past will continue to like similar products in the future.
# 
# #### Content-Based System
# 
# A content-based recommender system recommends users similar items that the user has liked previously or is currently interested in. A content-based recommender system uses the data collected and then builds a user profile, and then the profile is used to make personalized suggestions to the user. Content-based systems are less affected by the cold start problem, new users, or items that don't have a track record since users can be described by their characteristics.
# 
# ![rec-systems.png](attachment:rec-systems.png)
# 
# We built a hybrid system that combines the content-based and collaborative filtering systems. The goal of the hybrid system is to combine the strengths of both systems and to reduce the issue of the cold start problem typical with collaborative filtering.  

# In[1]:


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
ratings = pd.read_csv('/Users/chocz/Documents/Recommender_system/ratings.csv')
users = pd.read_csv('/Users/chocz/Documents/Recommender_system/tags.csv')
movies = pd.read_csv('/Users/chocz/Documents/Recommender_system/movies.csv')


# In[2]:


movies.tail()
movies['genres'] = movies['genres'].str.replace('|',' ')


# In[3]:


len(movies.movieId.unique())


# In[4]:


len(ratings.movieId.unique())


# In[5]:


ratings_f = ratings.groupby('userId').filter(lambda x: len(x) >= 55)

movie_list_rating = ratings_f.movieId.unique().tolist()


# In[6]:


len(ratings_f.movieId.unique())/len(movies.movieId.unique()) * 100


# In[7]:


len(ratings_f.userId.unique())/len(ratings.userId.unique()) * 100


# In[8]:


movies = movies[movies.movieId.isin(movie_list_rating)]


# In[9]:


movies.head()


# In[10]:


Mapping_file = dict(zip(movies.title.tolist(), movies.movieId.tolist()))


# In[11]:


users.drop(['timestamp'],1, inplace=True)
ratings_f.drop(['timestamp'],1, inplace=True)


# In[12]:


mixed = pd.merge(movies, users, on='movieId', how='left')
mixed.head()


# In[13]:


mixed.fillna("", inplace=True)
mixed = pd.DataFrame(mixed.groupby('movieId')['tag'].apply(
                                          lambda x: "%s" % ' '.join(x)))
Final = pd.merge(movies, mixed, on='movieId', how='left')
Final ['metadata'] = Final[['tag', 'genres']].apply(
                                          lambda x: ' '.join(x), axis = 1)
Final[['movieId','title','metadata']].head(3)


# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(Final['metadata'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=Final.index.tolist())
print(tfidf_df.shape)


# In[15]:


# SVD
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=200)
latent_matrix = svd.fit_transform(tfidf_df)
# plot variance explained
explained = svd.explained_variance_ratio_.cumsum()
plt.plot(explained, '.-', ms = 16, color='red')
plt.xlabel('Singular value components', fontsize= 12)
plt.ylabel('Cumulative percent of variance', fontsize=12)        
plt.show()


# In[16]:


n = 25
latent_matrix_1_df = pd.DataFrame(latent_matrix[:,0:n], index=Final.title.tolist())


# In[17]:


latent_matrix.shape


# In[18]:


ratings_f.head()


# In[19]:


ratings_f1 = pd.merge(movies[['movieId']], ratings_f, on="movieId", how="right")


# In[20]:


ratings_f2 = ratings_f1.pivot(index = 'movieId', columns ='userId', values = 'rating').fillna(0)


# In[21]:


ratings_f2.head()


# In[22]:


len(ratings_f.movieId.unique())


# In[23]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=25)
latent_matrix_2 = svd.fit_transform(ratings_f2)
latent_matrix_2_df = pd.DataFrame(
                             latent_matrix_2,
                             index=Final.title.tolist())


# In[24]:


explained = svd.explained_variance_ratio_.cumsum()
plt.plot(explained, '.-', ms = 16, color='red')
plt.xlabel('Singular value components', fontsize= 12)
plt.ylabel('Cumulative percent of variance', fontsize=12)        
plt.show()


# In[25]:


from sklearn.metrics.pairwise import cosine_similarity
# take the latent vectors for a selected movie from both content 
# and collaborative matrixes
#you can enter any movie in the dataset and it will return 20 recommended movies from the hybrid system
a_1 = np.array(latent_matrix_1_df.loc['Toy Story (1995)']).reshape(1, -1)
a_2 = np.array(latent_matrix_2_df.loc['Toy Story (1995)']).reshape(1, -1)

# calculate the similarity of this movie with the others in the list
score_1 = cosine_similarity(latent_matrix_1_df, a_1).reshape(-1)
score_2 = cosine_similarity(latent_matrix_2_df, a_2).reshape(-1)

# hybrid system is simply the average of content-based and collaborative filtering
hybrid = ((score_1 + score_2)/2.0)

# form a data frame of similar movies 
dictDf = {'content': score_1 , 'collaborative': score_2, 'hybrid': hybrid} 
similar = pd.DataFrame(dictDf, index = latent_matrix_1_df.index )

#sort it by hybrid score
similar.sort_values('hybrid', ascending=False, inplace=True)

#Show the top 20 recommended movies based on hybrid system
similar[1:].head(20)


# In[33]:


from sklearn.metrics.pairwise import cosine_similarity
# take the latent vectors for a selected movie from both content 
# and collaborative matrixes
a_1 = np.array(latent_matrix_1_df.loc['Interstellar (2014)']).reshape(1, -1)
a_2 = np.array(latent_matrix_2_df.loc['Interstellar (2014)']).reshape(1, -1)

# calculate the similarity of this movie with the others in the list
score_1 = cosine_similarity(latent_matrix_1_df, a_1).reshape(-1)
score_2 = cosine_similarity(latent_matrix_2_df, a_2).reshape(-1)

# hybrid system is simply the average of content-based and collaborative filtering
hybrid = ((score_1 + score_2)/2.0)

# form a data frame of similar movies 
dictDf = {'content': score_1 , 'collaborative': score_2, 'hybrid': hybrid} 
similar = pd.DataFrame(dictDf, index = latent_matrix_1_df.index )

#sort it by hybrid score
similar.sort_values('hybrid', ascending=False, inplace=True)

#Show the top 10 recommended movies based on hybrid system
similar[1:].head(20)


# In[27]:


from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split


# In[28]:


reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_f[['userId','movieId','rating']], reader)

#train-test split: 75% train and 25% test
trainset, testset = train_test_split(data, test_size=.25)
algorithm = SVD()
algorithm.fit(trainset)
predictions = algorithm.test(testset)

#RMSE on test set
accuracy.rmse(predictions)


# In[29]:


def pred_user_rating(ui):
    if ui in ratings_f.userId.unique():
        ui_list = ratings_f[ratings_f.userId == ui].movieId.tolist()
        d = {k: v for k,v in Mapping_file.items() if not v in ui_list}        
        predictedL = []
        for i, j in d.items():     
            predicted = algorithm.predict(ui, j)
            predictedL.append((i, predicted[3])) 
        pdf = pd.DataFrame(predictedL, columns = ['movies', 'ratings'])
        pdf.sort_values('ratings', ascending=False, inplace=True)  
        pdf.set_index('movies', inplace=True)    
        return pdf.head(10)        
    else:
        print("User Id does not exist.")
        return None


# In[30]:


user_id = 8
pred_user_rating(user_id)


# In[32]:
import joblib
pickle.dump(hybrid, open('model.pkl','wb'))


# In[ ]:

model=pickle.load(open('model.pkl','rb'))

model = joblib.load('model.pkl')

# In[ ]:


model.predict('Avatar')

