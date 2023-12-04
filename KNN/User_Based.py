from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import zipfile
import sys, os
from recsys.datasets import ml1m
from recsys.preprocessing import ids_encoder


# Uncomment the below lines if recsys is not available in the same dir

# import shutil
# !wget https://github.com/nzhinusoftcm/review-on-collaborative-filtering/raw/master/recsys.zip    
# shutil.unpack_archive('recsys.zip') 


ratings, movies = ml1m.load()

# create the encoder
ratings, uencoder, iencoder = ids_encoder(ratings)


# Transform rating dataframe to matrix
def ratings_matrix(ratings):    
    return csr_matrix(pd.crosstab(ratings.userid, ratings.itemid, ratings.rating, aggfunc=sum).fillna(0).values)    

R = ratings_matrix(ratings)


def create_model(rating_matrix, metric):
    """
    - create the nearest neighbors model with the corresponding similarity metric
    - fit the model
    """
    model = NearestNeighbors(metric=metric, n_neighbors=21, algorithm='brute')
    model.fit(rating_matrix)    
    return model


def nearest_neighbors(rating_matrix, model):
    """    
    :param rating_matrix : rating matrix of shape (nb_users, nb_items)
    :param model : nearest neighbors model    
    :return
        - similarities : distances of the neighbors from the referenced user
        - neighbors : neighbors of the referenced user in decreasing order of similarities
    """    
    similarities, neighbors = model.kneighbors(rating_matrix)        
    return similarities[:, 1:], neighbors[:, 1:]


model = create_model(rating_matrix=R, metric='cosine') # we can also use the 'euclidian' distance
similarities, neighbors = nearest_neighbors(R, model)


def find_candidate_items(userid):
    """
    Find candidate items for an active user
    
    :param userid : active user
    :param neighbors : users similar to the active user        
    :return candidates : top 30 of candidate items
    """
    user_neighbors = neighbors[userid]
    activities = ratings.loc[ratings.userid.isin(user_neighbors)]
    
    # sort items in decreasing order of frequency
    frequency = activities.groupby('itemid')['rating'].count().reset_index(name='count').sort_values(['count'],ascending=False)
    Gu_items = frequency.itemid
    active_items = ratings.loc[ratings.userid == userid].itemid.to_list()
    candidates = np.setdiff1d(Gu_items, active_items, assume_unique=True)[:30]
        
    return candidates


# mean ratings for each user
mean = ratings.groupby(by='userid', as_index=False)['rating'].mean()
mean_ratings = pd.merge(ratings, mean, suffixes=('','_mean'), on='userid')

# normalized ratings for each items
mean_ratings['norm_rating'] = mean_ratings['rating'] - mean_ratings['rating_mean']
mean = mean.to_numpy()[:, 1]
np_ratings = mean_ratings.to_numpy()


def predict(userid, itemid):
    """
    predict what score userid would have given to itemid.
    
    :param
        - userid : user id for which we want to make prediction
        - itemid : item id on which we want to make prediction
        
    :return
        - r_hat : predicted rating of user userid on item itemid
    """
    user_similarities = similarities[userid]
    user_neighbors = neighbors[userid]
    # get mean rating of user userid
    user_mean = mean[userid]
    
    # find users who rated item 'itemid'
    iratings = np_ratings[np_ratings[:, 1].astype('int') == itemid]
    
    # find similar users to 'userid' who rated item 'itemid'
    suri = iratings[np.isin(iratings[:, 0], user_neighbors)]
    
    # similar users who rated current item (surci)
    normalized_ratings = suri[:,4]
    indexes = [np.where(user_neighbors == uid)[0][0] for uid in suri[:, 0].astype('int')]
    sims = user_similarities[indexes]
    
    num = np.dot(normalized_ratings, sims)
    den = np.sum(np.abs(sims))
    
    if num == 0 or den == 0:
        return user_mean
    
    r_hat = user_mean + np.dot(normalized_ratings, sims) / np.sum(np.abs(sims))
    
    return r_hat


def user2userPredictions(userid, pred_path):
    """
    Make rating prediction for the active user on each candidate item and save in file prediction.csv
    
    :param
        - userid : id of the active user
        - pred_path : where to save predictions
    """    
    # find candidate items for the active user
    candidates = find_candidate_items(userid)
    
    # loop over candidates items to make predictions
    for itemid in candidates:
        
        # prediction for userid on itemid
        r_hat = predict(userid, itemid)
        
        # save predictions
        with open(pred_path, 'a+') as file:
            line = '{},{},{}\n'.format(userid, itemid, r_hat)
            file.write(line)



def user2userCF():
    """
    Make predictions for each user in the database.    
    """
    # get list of users in the database
    users = ratings.userid.unique()
    
    def _progress(count):
        sys.stdout.write('\rRating predictions. Progress status : %.1f%%' % (float(count/len(users))*100.0))
        sys.stdout.flush()
    
    saved_predictions = 'predictions.csv'    
    if os.path.exists(saved_predictions):
        os.remove(saved_predictions)
    
    for count, userid in enumerate(users):        
        # make rating predictions for the current user
        user2userPredictions(userid, saved_predictions)
        _progress(count)


user2userCF()
print("\n")


def user2userRecommendation(userid):
    """
    """
    # encode the userid
    uid = uencoder.transform([userid])[0]
    saved_predictions = 'predictions.csv'
    
    predictions = pd.read_csv(saved_predictions, sep=',', names=['userid', 'itemid', 'predicted_rating'])
    predictions = predictions[predictions.userid==uid]
    List = predictions.sort_values(by=['predicted_rating'], ascending=False)
    
    List.userid = uencoder.inverse_transform(List.userid.tolist())
    List.itemid = iencoder.inverse_transform(List.itemid.tolist())
    
    List = pd.merge(List, movies, on='itemid', how='inner')
    
    return List


# predictions = pd.read_csv('predictions.csv', sep=',', names=['userid', 'itemid', 'predicted_rating'])
# df = pd.merge(predictions, movies, on='itemid', how='inner')
# df.to_csv('predictions_with_genres.csv', index=False)


user2userRecommendation(212)


def calculate_ndcg(actual, recommended):
    dcg = 0
    for i in range(len(recommended)):
        item = recommended[i]
        if item in actual:
            dcg += 1 / np.log2(i + 2)  # Index starts at 0, so add 2
    idcg = sum(1 / np.log2(i + 2) for i in range(len(actual)))
    ndcg = dcg / idcg
    return ndcg


from recsys.preprocessing import train_test_split, get_examples
from sklearn.metrics import precision_score, recall_score

# get examples as tuples of userids and itemids and labels from normalize ratings
raw_examples, raw_labels = get_examples(ratings, labels_column='rating')

# train test split
(x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)

def evaluate(x_test, y_test):
    print('Evaluate the model on {} test data ...'.format(x_test.shape[0]))
    preds = list(predict(u,i) for (u,i) in x_test)
    mae = np.sum(np.absolute(y_test - np.array(preds))) / x_test.shape[0]
    print('\nMAE :', mae)

    precision = precision_score(y_test, list(map(int, preds)), average='micro')
    print(f"Precision: {precision:.4f}")
    
    recall = recall_score(y_test, list(map(int, preds)), average='micro')
    print(f"Recall: {recall:.4f}")
    
    ndcg = calculate_ndcg(y_test, list(map(int, preds)))
    print(f"Ndcg: {ndcg:.4f}")
    
    return mae


evaluate(x_test, y_test)


from recsys.memories.UserToUser import UserToUser

# load ml1m ratings
ratings, movies = ml1m.load()

# prepare data
ratings, uencoder, iencoder = ids_encoder(ratings)

# get examples as tuples of userids and itemids and labels from normalize ratings
raw_examples, raw_labels = get_examples(ratings, labels_column='rating')

# train test split
(x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)


# create the user-based CF
usertouser = UserToUser(ratings, movies, metric='cosine')


# evaluate the user-based CF on the ml1m test data
usertouser.evaluate(x_test, y_test)


from recsys.datasets import ml1m
from recsys.preprocessing import ids_encoder, get_examples, train_test_split
from recsys.memories.UserToUser import UserToUser

# load ml1m ratings
ratings, movies = ml1m.load()

# prepare data
ratings, uencoder, iencoder = ids_encoder(ratings)

# get examples as tuples of userids and itemids and labels from normalize ratings
raw_examples, raw_labels = get_examples(ratings, labels_column='rating')

# train test split
(x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)

# create the user-based CF
usertouser = UserToUser(ratings, movies, k=20, metric='cosine')

# evaluate the user-based CF on the ml1m test data
print("==========================")
usertouser.evaluate(x_test, y_test)