import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import matplotlib.pyplot as plt
#from sklearn.externals import joblib


ratings = pd.read_csv('/Users/chocz/Documents/Recommender_system/ratings.csv')
users = pd.read_csv('/Users/chocz/Documents/Recommender_system/tags.csv')
movies = pd.read_csv('/Users/chocz/Documents/Recommender_system/movies.csv')

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train')
def train():
        
    ratings = pd.read_csv('/Users/chocz/Documents/Recommender_system/ratings.csv')
    users = pd.read_csv('/Users/chocz/Documents/Recommender_system/tags.csv')
    movies = pd.read_csv('/Users/chocz/Documents/Recommender_system/movies.csv')

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
    
    with open('hybrid.pickle','wb') as pickle_out:
        pickle.dump(hybrid,pickle_out)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    movie = request.form.get("title", False)  
    movies.tail()
    movies['genres'] = movies['genres'].str.replace('|',' ')
    with open('hybrid.pickle','rb') as pickle_in:  
        model=pickle.load(pickle_in)
        recommendations=model.predict(movie)
        # form a data frame of similar movies 
    dictDf = {'hybrid': movie} 
    similar = pd.DataFrame(dictDf, index = latent_matrix_1_df.index )

#sort it by hybrid score
    similar.sort_values('hybrid', ascending=False, inplace=True)

#Show the top 10 recommended movies based on hybrid system
    similar[1:].head(20)
    return render_template('index.html', prediction_text='The recommended movies are ${}'.format(recommendations))

if __name__ == "__main__":
    app.run(debug=True)
