#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD
import ast 
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import warnings; warnings.simplefilter('ignore')


# In[3]:


ratings=pd.read_csv('/Users/chocz/Documents/ratings_small.csv')# Contains ratings of the movies
metadata=pd.read_csv('/Users/chocz/Documents/movies_metadata.csv')# Contains information about the movies, eg:genre, busget, language etc
links=pd.read_csv('/Users/chocz/Documents/links_small.csv')# Contains IMDB and TMDB IDs of all movies
keywords=pd.read_csv('/Users/chocz/Documents/keywords.csv')# Contains information about the movie and key words defining the movie
credits=pd.read_csv('/Users/chocz/Documents/credits.csv')# Contains names of cast and crew


# In[4]:


column_names = ['user_id', 'item_id']
titles = pd.read_csv('/Users/chocz/Documents/u.item.csv',sep='/t',names=column_names) 


# In[5]:


titles.head()


# In[6]:


titles=titles.user_id.apply(lambda x: pd.Series(str(x).split("|")))


# In[7]:


titles.head()


# In[8]:


titles=titles.iloc[:,[0,1]]


# In[9]:


titles.head()


# In[10]:


titles[['item_id','name']]=titles.iloc[:,[0,1]]
titles=titles.iloc[:,[2,3]]


# In[11]:


titles.head()


# In[12]:


ratings['item_Id']=ratings['movieId']


# In[13]:


ratings.head()


# In[14]:


len(ratings.movieId)


# In[15]:


credits.shape


# In[16]:


len(credits.id.unique())


# In[17]:


credits.columns


# In[18]:


credits.info()


# In[19]:


credits.head()


# In[20]:


credits=credits.iloc[:,0:3]
credits.head()


# In[21]:


links.head()


# cast: Information about the cast. Name of actor, gender and it's character name in movie
# 
# crew: Information about crew members. eg: Director, Editor etc
# 
# id:   It's movie ID given by TMDb
# 

# In[22]:


links.columns


# In[23]:


links['item_Id']=links['movieId']
links.info()


# In[24]:


keywords.head()


# movieId: Serial number for movie
# 
# imdbId: Movie id given by IMDb
# 
# tmdbId: Movie id given by TMDb 
# 

# In[25]:


keywords.columns


# In[26]:


keywords.info()


# id: It's movie ID given by TMDb
# 
# Keywords: Tags/keywords for the movie. It list of tags/keywords
# 

# In[27]:


metadata.head()


# In[28]:


metadata.iloc[0:2].transpose()


# In[29]:


metadata.columns


# adult: Indicates if the movie is X-Rated or Adult.
# 
# belongs_to_collection: A stringified dictionary that gives information on the movie series the particular film belongs to.
# 
# budget: The budget of the movie in dollars.
# 
# genres: A stringified list of dictionaries that list out all the genres associated with the movie.
# 
# homepage: The Official Homepage of the move.
# 
# id: The ID of the movie.
# 
# imdb_id: The IMDB ID of the movie.
# 
# original_language: The language in which the movie was originally shot in.
# 
# original_title: The original title of the movie.
# 
# overview: A brief blurb of the movie.
# 
# popularity: The Popularity Score assigned by TMDB.
# 
# poster_path: The URL of the poster image.
# 
# production_companies: Production companies involved with the making of the movie.
# 
# production_countries: A stringified list of countries where the movie was shot/produced in.
# 
# release_date: Theatrical Release Date of the movie.
# 
# revenue: The total revenue of the movie in dollars.
# 
# runtime: The runtime of the movie in minutes.
# 
# spoken_languages: A stringified list of spoken languages in the film.
# status: The status of the movie (Released, To Be Released, Announced, etc.)
# 
# tagline: The tagline of the movie.
# 
# title: The Official Title of the movie.
# 
# video: Indicates if there is a video present of the movie with TMDB.
# 
# vote_average: The average rating of the movie.
# 
# vote_count: The number of votes by users, as counted by TMDB.
# 

# In[30]:


metadata.info()


# In[31]:


ratings.head()


# In[32]:


len(ratings.userId.unique())


# In[33]:


ratings.head()


# In[34]:


ratings[['item_Id']]=ratings.item_Id.astype(str)
ratings.info()


# In[35]:


ratings.groupby('movieId')['rating'].count().sort_values(ascending=False).head()


# In[36]:


#ratings=pd.read_csv('/Users/chocz/Documents/ml-latest-small/ratings.csv')# Contains ratings of the movies
#metadata=pd.read_csv('/Users/chocz/Documents/movies_metadata.csv')# Contains information about the movies, eg:genre, busget, language etc
#links=pd.read_csv('/Users/chocz/Documents/ml-latest-small/links.csv')# Contains IMDB and TMDB IDs of all movies
#keywords=pd.read_csv('/Users/chocz/Documents/ml-latest-small/tags.csv')# Contains information about the movie and key words defining the movie
#credits=pd.read_csv('/Users/chocz/Documents/credits.csv')# Contains names of cast and crew


# In[37]:


len(titles.name.unique())


# In[38]:


titles[titles.name.duplicated(keep=False)].sort_values(by='name')


# In[39]:


titles['item_Id']=titles['item_id']


# In[40]:


df = pd.merge(ratings, titles,on='item_Id')
df=df.drop(columns=['movieId','timestamp'])
df.head()


# In[41]:


df=df.drop(columns=['item_id'])


# In[42]:


df.describe()


# In[43]:


print(df[df.name=="Body Snatchers (1993)"])


# In[44]:


df.groupby('name')['rating'].mean()


# In[45]:


df.groupby('name')['rating'].count().sort_values(ascending=False).head()


# In[46]:


ratings_df = pd.DataFrame(df.groupby('name')['rating'].mean())
ratings_df.head()


# In[47]:


ratings_df.rename(columns={'rating': 'average_rating'}, inplace=True)
ratings_df.head(3)


# In[48]:


ratings_df['num_of_ratings'] = pd.DataFrame(df.groupby('name')['rating'].count())
ratings_df.head()


# In[49]:


ratings_df.info()


# In[50]:


from matplotlib import pyplot as plt
plt.figure(figsize=[10,6]) 
ratings_df['num_of_ratings'].hist(bins=50)
plt.xlabel('number of ratings')
plt.ylabel('number of films with that many ratings')


# In[51]:


plt.figure(figsize=[10,6])
ratings_df.average_rating.hist(bins=50)
plt.xlabel('rating (number of stars)')
plt.ylabel('number of films with that rating')


# In[52]:


import seaborn as sns
sns.jointplot(x='average_rating',y='num_of_ratings', data=ratings_df, alpha=0.5)


# In[53]:


ratings_df.sort_values('num_of_ratings',ascending=False).head(10)


# Recommender System

# In[54]:


metadata['genres'] = metadata['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i[
    'name'] for i in x] if isinstance(x, list) else [])


# In[55]:


vote_counts = metadata[metadata['vote_count'].notnull()]['vote_count'].astype('int')

vote_averages = metadata[metadata['vote_average'].notnull()]['vote_average'].astype('int')

C = vote_averages.mean()
C


# In[56]:


m = vote_counts.quantile(0.95)
m


# In[57]:


metadata['year'] = pd.to_datetime(metadata['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[58]:


qualified = metadata[(metadata['vote_count'] >= m) & 
               (metadata['vote_count'].notnull()) & 
               (metadata['vote_average'].notnull())][['title', 
                                                'year', 
                                                'vote_count', 
                                                'vote_average', 
                                                'popularity', 
                                                'genres']]

qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape


# In[59]:


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)


# In[60]:


qualified['wr'] = qualified.apply(weighted_rating, axis=1)


# In[61]:


qualified = qualified.sort_values('wr', ascending=False).head(250)


# In[62]:


qualified.head(15)


# In[63]:


s = metadata.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = metadata.drop('genres', axis=1).join(s)
gen_md.head(3).transpose()


# In[64]:


def build_chart(genre, percentile=0.8):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & 
                   (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: 
                        (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C),
                        axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified


# In[65]:


build_chart('Romance').head(15)


# In[66]:


links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[67]:


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[68]:


metadata['id'] = metadata['id'].apply(convert_int)
metadata[metadata['id'].isnull()]


# In[69]:


metadata = metadata.drop([19730, 29503, 35587])


# In[70]:


metadata['id'] = metadata['id'].astype('int')


# In[71]:


smd = metadata[metadata['id'].isin(links)]
smd.shape


# In[72]:


smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')


# In[73]:


tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])


# In[74]:


tfidf_matrix.shape


# In[75]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[0]


# In[76]:


smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


# In[77]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[78]:


get_recommendations('The Godfather').head(10)


# In[79]:


get_recommendations('The Dark Knight').head(10)


# Content based Recommender System using Metadata

# In[80]:


keywords['id'] = keywords['id'].astype('int')
credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
credits['id'] = credits['id']
metadata['id'] = metadata['id'].astype('int')


# In[81]:


metadata.shape


# In[82]:


metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')


# In[83]:


credits.info()


# In[84]:


smd = metadata[metadata['id'].isin(links)]
smd.shape


# In[85]:


smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))


# In[86]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[87]:


smd['director'] = smd['crew'].apply(get_director)
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[88]:


smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x,x, x])


# In[89]:


s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s[:5]


# In[90]:


s = s[s > 1]


# In[91]:


stemmer = SnowballStemmer('english')


# In[92]:


def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words


# In[93]:


smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


# In[94]:


smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))


# In[95]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])


# In[96]:


cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[97]:


smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


# In[98]:


get_recommendations('The Dark Knight').head(10)


# In[99]:


get_recommendations('Inception').head(10)


# In[100]:


def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & 
                       (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified


# In[101]:


improved_recommendations('The Dark Knight')


# In[102]:


reader = Reader()


# In[103]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


# In[104]:


from surprise.model_selection import cross_validate
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'],cv=5)


# In[105]:


from surprise.model_selection import train_test_split
from surprise import accuracy
trainset, testset = train_test_split(data, test_size=.25)
svd.fit(trainset)
predictions = svd.test(testset)
accuracy.rmse(predictions)


# In[106]:


ratings[ratings['userId'] == 1]


# In[107]:


svd.predict(1, 302)


# In[108]:


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[109]:


id_map = pd.read_csv('/Users/chocz/Documents/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')


# In[110]:


indices_map = id_map.set_index('id')


# In[119]:


def hybrid(title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'release_date', 'id']]
    return movies.head(10)


# In[121]:


hybrid('The Dark Knight')


# In[ ]:





# In[ ]:




