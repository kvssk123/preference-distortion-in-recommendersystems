import torch
import torch.nn.functional as F
# from torch import linalg as LA

import time
import numpy as np
import pickle
import argparse
import pandas as pd
import utility
from scipy.sparse import csr_matrix, rand as sprand
from tqdm import tqdm

np.random.seed(0)
torch.manual_seed(0)


class MF(torch.nn.Module):
    def __init__(self, arguments, train_df, train_like, test_like):
        super(MF, self).__init__()

        self.num_users = arguments['num_user']
        self.num_items = arguments['num_item']
        self.learning_rate = arguments['learning_rate']
        self.epochs = arguments['epoch']
        self.display = arguments['display']
        self.regularization = arguments['reg']
        self.hidden = arguments['hidden']
        self.neg_sampling = arguments['neg']
        self.data = arguments['data']
        self.batch_size = arguments['bs']

        self.train_df = train_df
        self.train_like = train_like
        self.test_like = test_like

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")

        print('******************** MF ********************')

        self.user_factors = torch.nn.Embedding(self.num_users, self.hidden)
        self.user_factors.weight.data.uniform_(-0.05, 0.05)
        self.item_factors = torch.nn.Embedding(self.num_items, self.hidden)
        self.item_factors.weight.data.uniform_(-0.05, 0.05)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.regularization)
        self.loss_function = torch.nn.MSELoss()

        print('********************* MF Initialization Done *********************')

        self.to(self.device)

    def forward(self, user, item):
        # Get the dot product per row
        u = self.user_factors(user)
        v = self.item_factors(item)
        x = (u * v).sum(axis=1)

        return x

    def train_model(self, itr):
        self.train()
        epoch_cost = 0.
        self.user_list, self.item_list, self.label_list = utility.negative_sampling(self.num_users, self.num_items,
                                                                                    self.train_df['userId'].values,
                                                                                    self.train_df['itemId'].values,
                                                                                    self.neg_sampling)
        start_time = time.time() * 1000.0
        num_batch = int(len(self.user_list) / float(self.batch_size)) + 1
        random_idx = np.random.permutation(len(self.user_list))

        for i in tqdm(range(num_batch)):
            batch_idx = None
            if i == num_batch - 1:
                batch_idx = random_idx[i * self.batch_size:]
            elif i < num_batch - 1:
                batch_idx = random_idx[(i * self.batch_size):((i + 1) * self.batch_size)]

            tmp_cost = self.train_batch(self.user_list[batch_idx], self.item_list[batch_idx],
                                                              self.label_list[batch_idx])

            epoch_cost += tmp_cost

        print("Training //", "Epoch %d //" % itr, " Total cost = {:.5f}".format(epoch_cost),
              "Training time : %d ms" % (time.time() * 1000.0 - start_time))

    def train_batch(self, user_input, item_input, label_input):
        # reset gradients
        self.optimizer.zero_grad()

        users = torch.Tensor(user_input).long().to(self.device)
        items = torch.Tensor(item_input).long().to(self.device)
        labels = torch.Tensor(label_input).float().to(self.device)

        y_hat = self.forward(users, items)
        loss = self.loss_function(y_hat, labels)

        # backpropagate
        loss.backward()
        # update
        self.optimizer.step()

        return loss.item()


    def test_model(self, itr):  # calculate the cost and rmse of testing set in each epoch
        self.eval()

        start_time = time.time() * 1000.0
        P, Q = self.user_factors.weight, self.item_factors.weight
        P = P.detach().numpy()
        Q = Q.detach().numpy()
        Rec = np.matmul(P, Q.T)

        precision, recall, ndcg = utility.MP_test_model_all(Rec, self.test_like, self.train_like, n_workers=10)

        print("Testing //", "Epoch %d //" % itr,
              "Accuracy Testing time : %d ms" % (time.time() * 1000.0 - start_time))
        print("=" * 100)
        print("test_model_ndcg: ", np.mean(ndcg))
        return np.mean(ndcg)


    def run(self):
        best_metric = -1
        best_itr = 0
        for epoch_itr in range(1, self.epochs + 1):
            self.train_model(epoch_itr)
            if epoch_itr % self.display == 0:
                cur_metric = self.test_model(epoch_itr)
                if cur_metric > best_metric:
                    print("mertric: ", best_metric)
                    best_metric = cur_metric
                    best_itr = epoch_itr
                    self.make_records(epoch_itr)
                elif epoch_itr - best_itr >= 30:
                    break

    def make_records(self, itr):  # record all the results' details into files
        P, Q = self.user_factors.weight, self.item_factors.weight
        P = P.detach().cpu().numpy()
        Q = Q.detach().cpu().numpy()
        # with open('./' + self.args.data + '/P_MF_' + str(self.reg) + '.npy', "wb") as f:
        #     np.save(f, P)
        # with open('./' + self.args.data + '/Q_MF_' + str(self.reg) + '.npy', "wb") as f:
        #     np.save(f, Q)
        return P, Q


if __name__ == '__main__':
    args = {
        'epoch': 100,
        'display': 1,
        'learning_rate': 0.001,
        'reg': 0.0001,
        'hidden': 64,
        'neg': 2,
        'bs': 1024,
        'data': 'ML1M'
    }

    with open('MF_RMSE/ML1M/info.pkl', 'rb') as f:
        info = pickle.load(f)
        args['num_user'] = info['num_user']
        args['num_item'] = info['num_item']
    print(info)

    train_like = list(np.load('MF_RMSE/ML1M/user_train_like.npy', allow_pickle=True))
    # test_like = list(np.load('MF_RMSE/ML1M/user_vali_like.npy', allow_pickle=True))
    test_like = list(np.load('MF_RMSE/ML1M/user_test_like.npy', allow_pickle=True))
    train_df = pd.read_csv('MF_RMSE/ML1M/train_df.csv')

    model = MF(args, train_df, train_like, test_like)
    # model.run()
    
    # ###
    model_file_path = 'mf_model.pth'

    # # Save the model to the specified file
    # torch.save(model.state_dict(), model_file_path)

    model.load_state_dict(torch.load(model_file_path))
    model.eval()  # Set the model to evaluation mode

    # Load the 'movies.csv' file
    movies_df = pd.read_csv('MF_RMSE/ML1M/movie_df.csv')

    users = pd.read_csv('MF_RMSE/ML1M/ratings.dat', sep = "::")

    # Create a list of user IDs
    user_ids = users.iloc[:, 0].unique()


    # Create an empty DataFrame to store recommendations
    all_recommendations = []

    # Let's say you want recommendations for user with ID 'user_id'.
    # user_id = 781  # Replace with the actual user ID for which you want recommendations

    for user_id in user_ids:
        # Create a list of item IDs that the user has not interacted with in the training data.
        # You can do this by excluding items that the user has already interacted with.
        train_user_items = train_df[train_df['userId'] == user_id]['itemId'].values
        all_item_ids = np.arange(args['num_item'])
        unseen_items = np.setdiff1d(all_item_ids, train_user_items)

        # Predict scores for the unseen items for the user
        user_tensor = torch.LongTensor([user_id]).to(model.device)
        unseen_items_tensor = torch.LongTensor(unseen_items).to(model.device)
        
        try:
            # Use the model to predict scores
            scores = model.forward(user_tensor, unseen_items_tensor)
            scores = scores.cpu().detach().numpy()

            # Sort the items by predicted scores in descending order to get recommendations
            recommended_item_ids = unseen_items[np.argsort(-scores)]

            # You can now take the top N items as recommendations, e.g., top 10 items.
            top_N = 30
            recommendations = recommended_item_ids[:top_N]

            print(f"Recommendations for user {user_id}")

            # Get movie details for the recommended items
            recommended_movies = movies_df[movies_df['itemId'].isin(recommendations)]

            # Store recommendations in a list
            for movie_id in recommendations:
                movie_details = recommended_movies[recommended_movies['itemId'] == movie_id]
                if not movie_details.empty:
                    movie_name = movie_details.iloc[0]['title']
                    movie_genres = movie_details.iloc[0]['genres']
                    score = scores[unseen_items == movie_id][0]
                    all_recommendations.append([user_id, movie_id, movie_name, movie_genres, score])
        except:
            pass

    # Create a DataFrame from the recommendations list
    recommendations_df = pd.DataFrame(all_recommendations, columns=['user_id', 'movie_id', 'movie_name', 'genres', 'score'])

    # Save the recommendations DataFrame to a CSV file
    recommendations_df.to_csv('user_recommendations.csv', index=False)