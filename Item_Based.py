from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import os
import sys
from recsys.datasets import ml1m, ml100k
from recsys.preprocessing import ids_encoder

# Uncomment the below lines if recsys is not available in the same dir

# import shutil
# !wget https://github.com/nzhinusoftcm/review-on-collaborative-filtering/raw/master/recsys.zip    
# shutil.unpack_archive('recsys.zip')


ratings, movies = ml1m.load()

# create the encoder
ratings, uencoder, iencoder = ids_encoder(ratings)


def normalize():
    # compute mean rating for each user
    mean = ratings.groupby(by='userid', as_index=False)['rating'].mean()
    norm_ratings = pd.merge(ratings, mean, suffixes=('','_mean'), on='userid')
    
    # normalize each rating by substracting the mean rating of the corresponding user
    norm_ratings['norm_rating'] = norm_ratings['rating'] - norm_ratings['rating_mean']
    return mean.to_numpy()[:, 1], norm_ratings


mean, norm_ratings = normalize()
np_ratings = norm_ratings.to_numpy()
norm_ratings.head()


def item_representation(ratings):    
    return csr_matrix(
        pd.crosstab(ratings.itemid, ratings.userid, ratings.norm_rating, aggfunc=sum).fillna(0).values
    )

R = item_representation(norm_ratings)


def create_model(rating_matrix, k=20, metric="cosine"):
    """
    :param R : numpy array of item representations
    :param k : number of nearest neighbors to return    
    :return model : our knn model
    """    
    model = NearestNeighbors(metric=metric, n_neighbors=k+1, algorithm='brute')
    model.fit(rating_matrix)    
    return model


def nearest_neighbors(rating_matrix, model):
    """
    compute the top n similar items for each item.    
    :param rating_matrix : items representations
    :param model : nearest neighbors model    
    :return similarities, neighbors
    """    
    similarities, neighbors = model.kneighbors(rating_matrix)    
    return similarities[:,1:], neighbors[:,1:]


def save_similarities(similarities, neighbors, dataset_name):    
    base_dir = 'recsys/weights/item2item'
    save_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)    
    similarities_file_name = os.path.join(save_dir, 'similarities.npy')
    neighbors_file_name = os.path.join(save_dir, 'neighbors.npy')    
    try:
        np.save(similarities_file_name, similarities)
        np.save(neighbors_file_name, neighbors)        
    except ValueError as error:
        print(f"An error occured when saving similarities, due to : \n ValueError : {error}")

        
def load_similarities(dataset_name, k=20):
    base_dir = 'recsys/weights/item2item'
    save_dir = os.path.join(base_dir, dataset_name)    
    similiraties_file = os.path.join(save_dir, 'similarities.npy')
    neighbors_file = os.path.join(save_dir, 'neighbors.npy')    
    similarities = np.load(similiraties_file)
    neighbors = np.load(neighbors_file)    
    return similarities[:,:k], neighbors[:,:k]


def cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def adjusted_cosine(np_ratings, nb_items, dataset_name):
    similarities = np.zeros(shape=(nb_items, nb_items))
    similarities.fill(-1)
    
    def _progress(count):
        sys.stdout.write('\rComputing similarities. Progress status : %.1f%%' % (float(count / nb_items)*100.0))
        sys.stdout.flush()
        
    items = sorted(ratings.itemid.unique())    
    for i in items[:-1]:
        for j in items[i+1:]:            
            scores = np_ratings[(np_ratings[:, 1] == i) | (np_ratings[:, 1] == j), :]
            vals, count = np.unique(scores[:,0], return_counts = True)
            scores = scores[np.isin(scores[:,0], vals[count > 1]),:]

            if scores.shape[0] > 2:
                x = scores[scores[:, 1].astype('int') == i, 4]
                y = scores[scores[:, 1].astype('int') == j, 4]
                w = cosine(x, y)

                similarities[i, j] = w
                similarities[j, i] = w
        _progress(i)
    _progress(nb_items)
    
    # get neighbors by their neighbors in decreasing order of similarities
    neighbors = np.flip(np.argsort(similarities), axis=1)
    
    # sort similarities in decreasing order
    similarities = np.flip(np.sort(similarities), axis=1)
    
    # save similarities to disk
    save_similarities(similarities, neighbors, dataset_name=dataset_name) 
    
    return similarities, neighbors


# metric : choose among [cosine, euclidean, adjusted_cosine]
metric = 'adjusted_cosine'

if metric == 'adjusted_cosine':
    similarities, neighbors = load_similarities('ml100k')
else:
    model = create_model(R, k=21, metric=metric)
    similarities, neighbors = nearest_neighbors(R, model)


print('neighbors shape : ', neighbors.shape)
print('similarities shape : ', similarities.shape)


def candidate_items(userid):
    """
    :param userid : user id for which we wish to find candidate items    
    :return : I_u, candidates
    """
    
    # 1. Finding the set I_u of items already rated by user userid
    I_u = np_ratings[np_ratings[:, 0] == userid]
    I_u = I_u[:, 1].astype('int')
    
    # 2. Taking the union of similar items for all items in I_u to form the set of candidate items
    c = set()
        
    for iid in I_u:
        try:
            # add the neighbors of item iid in the set of candidate items
            c.update(neighbors[iid])
        except IndexError:
            # Handle the IndexError and continue to the next item
            pass
    
    c = list(c)
    # 3. exclude from the set C all items in I_u.
    candidates = np.setdiff1d(c, I_u, assume_unique=True)
    
    return I_u, candidates


# Manually choose the user for testing
test_user = uencoder.transform([9])[0]
i_u, u_candidates = candidate_items(test_user)


print('number of items purchased by user 1 : ', len(i_u))
print('number of candidate items for user 1 : ', len(u_candidates))


def similarity_with_Iu(c, I_u):
    """
    compute similarity between an item c and a set of items I_u. For each item i in I_u, get similarity between 
    i and c, if c exists in the set of items similar to itemid.    
    :param c : itemid of a candidate item
    :param I_u : set of items already purchased by a given user    
    :return w : similarity between c and I_u
    """
    w = 0    
    for iid in I_u:
        try:
            # get similarity between itemid and c, if c is one of the k nearest neighbors of itemid
            if c in neighbors[iid]:
                w = w + similarities[iid, neighbors[iid] == c][0]
        except IndexError:
            # Handle the IndexError and continue to the next item in I_u
            pass
    return w


def rank_candidates(candidates, I_u):
    """
    rank candidate items according to their similarities with i_u    
    :param candidates : list of candidate items
    :param I_u : list of items purchased by the user    
    :return ranked_candidates : dataframe of candidate items, ranked in descending order of similarities with I_u
    """
    
    # list of candidate items mapped to their corresponding similarities to I_u
    sims = [similarity_with_Iu(c, I_u) for c in candidates]
    candidates = iencoder.inverse_transform(candidates)    
    mapping = list(zip(candidates, sims))
    
    ranked_candidates = sorted(mapping, key=lambda couple:couple[1], reverse=True)    
    return ranked_candidates


def topn_recommendation(userid, N=30):
    """
    Produce top-N recommendation for a given user    
    :param userid : user for which we produce top-N recommendation
    :param n : length of the top-N recommendation list    
    :return topn
    """
    # find candidate items
    I_u, candidates = candidate_items(userid)
    
    # rank candidate items according to their similarities with I_u
    ranked_candidates = rank_candidates(candidates, I_u)
    
    # get the first N row of ranked_candidates to build the top N recommendation list
    topn = pd.DataFrame(ranked_candidates[:N], columns=['itemid','similarity_with_Iu'])    
    topn = pd.merge(topn, movies, on='itemid', how='inner')    
    return topn


topn_recommendation(test_user)


def predict(userid, itemid):
    """
    Make rating prediction for user userid on item itemid    
    :param userid : id of the active user
    :param itemid : id of the item for which we are making prediction        
    :return r_hat : predicted rating
    """
    try:
        # Get items similar to item itemid with their corresponding similarities
        item_neighbors = neighbors[itemid]
        item_similarities = similarities[itemid]
    except IndexError:
        # Handle the IndexError when itemid is not found in neighbors
        return mean[userid]
   
    # get ratings of user with id userid
    uratings = np_ratings[np_ratings[:, 0].astype('int') == userid]
    
    # similar items rated by item the user of i
    siru = uratings[np.isin(uratings[:, 1], item_neighbors)]
    scores = siru[:, 2]
    indexes = [np.where(item_neighbors == iid)[0][0] for iid in siru[:,1].astype('int')]    
    sims = item_similarities[indexes]
    
    dot = np.dot(scores, sims)
    som = np.sum(np.abs(sims))

    if dot == 0 or som == 0:
        return mean[userid]
    
    return dot / som


def topn_prediction(userid):
    """
    :param userid : id of the active user    
    :return topn : initial topN recommendations returned by the function item2item_topN
    :return topn_predict : topN recommendations reordered according to rating predictions
    """
    # make top N recommendation for the active user
    topn = topn_recommendation(userid)
    
    # get list of items of the top N list
    itemids = topn.itemid.to_list()
    
    predictions = []
    
    # make prediction for each item in the top N list
    for itemid in itemids:
        r = predict(userid, itemid)
        
        predictions.append((itemid,r))
    
    predictions = pd.DataFrame(predictions, columns=['itemid','prediction'])
    
    # merge the predictions to topN_list and rearrange the list according to predictions
    topn_predict = pd.merge(topn, predictions, on='itemid', how='inner')
    topn_predict = topn_predict.sort_values(by=['prediction'], ascending=False)
    
    return topn, topn_predict


users = ratings.userid.unique()
users


# uncomment this to generate predictions for all users and save to the file.
# preds = []
# for user in users:
#     _, topn_predict = topn_prediction(userid=user)
#     preds.append(topn_predict)

# for i in range(len(users)):
#     # preds[i]['userid'] = users[i]
#     # preds[i]['prediction'] = preds[i].prediction.abs()
#     preds[i] = preds[i].sort_values(by=['prediction'], ascending=False)

# df = pd.concat(preds).reset_index(drop=True)

# df[['userid', 'itemid', 'similarity_with_Iu', 'title', 'genres', 'prediction']].to_csv('item_based_predictions_with_genres.csv', index = False)


print("Testing for this user_id:{}".format(test_user))


# In[31]:


topn, topn_predict = topn_prediction(userid=test_user)


# In[32]:


topn_predict


# As you will have noticed, the two lists are sorted in different ways. The second list is organized according to the predictions made for the user.
# 
# <b>Note</b>: When making predictions for user $u$ on item $i$, user $u$ may not have rated any of the $k$ most similar items to i. In this case, we consider the mean rating of $u$ as the predicted value.

# ## Evaluation with Mean Absolute Error

# In[33]:


def calculate_ndcg(actual, recommended):
    dcg = 0
    for i in range(len(recommended)):
        item = recommended[i]
        if item in actual:
            dcg += 1 / np.log2(i + 2)  # Index starts at 0, so add 2
    idcg = sum(1 / np.log2(i + 2) for i in range(len(actual)))
    ndcg = dcg / idcg
    return ndcg


# In[39]:


from recsys.preprocessing import train_test_split, get_examples

# get examples as tuples of userids and itemids and labels from normalize ratings
raw_examples, raw_labels = get_examples(ratings, labels_column='rating')

# train test split
(x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)

# def evaluate(x_test, y_test):
#     print('Evaluate the model on {} test data ...'.format(x_test.shape[0]))
#     preds = list(predict(u,i) for (u,i) in x_test)
#     mae = np.sum(np.absolute(y_test - np.array(preds))) / x_test.shape[0]
#     print('\nMAE :', mae)
#     return mae

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
    
    kl_divergence = np.sum(y_test * np.log(y_test / preds))
    
    print(f"KL divergence: {kl_divergence:.4f}")
    
    return mae


evaluate(x_test, y_test)


from recsys.memories.ItemToItem import ItemToItem
from recsys.preprocessing import ids_encoder, train_test_split, get_examples
from recsys.datasets import ml1m

# load data
ratings, movies = ml1m.load()

# prepare data
ratings, uencoder, iencoder = ids_encoder(ratings)

# get examples as tuples of userids and itemids and labels from normalize ratings
raw_examples, raw_labels = get_examples(ratings, labels_column='rating')

# train test split
(x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)

# create the Item-based CF
item2item = ItemToItem(ratings, movies, k=20, metric='cosine', dataset_name='ml1m')

# evaluate the algorithm on test dataset
item2item.evaluate(x_test, y_test)