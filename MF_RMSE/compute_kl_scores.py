import pandas as pd
from recsys.datasets import ml1m


def compute_genre_distr(items_df):
    """Compute the genre distribution for a given DataFrame of items."""
    distr = {}

    for index, row in items_df.iterrows():
        genre = row['genres']
        score = row['rating']
        # score = row['predicted_rating']
        # score = row['score']

        genre_score = distr.get(genre, 0.)
        distr[genre] = genre_score + score

    # normalize the summed-up probability so it sums up to 1
    # round it to three decimal places for added precision
    for genre, genre_score in distr.items():
        normed_genre_score = round(genre_score / len(items_df), 3)
        distr[genre] = normed_genre_score

    return distr


def replace_zero_with_small_value(input_dict, small_value=0.0001):
    """Replace all occurrences of 0 with a small_value in a dictionary."""
    return {key: small_value if value == 0 else value for key, value in input_dict.items()}


def compute_kl_divergence(interacted_distr, reco_distr, alpha=0.01):
    """
    KL (p || q), the lower the better.

    alpha is not really a tuning parameter, it's just there to make the
    computation more numerically stable.
    """
    kl_div = 0.
    for genre, score in interacted_distr.items():
        reco_score = reco_distr.get(genre, 0.)
        reco_score = (1 - alpha) * reco_score + alpha * score
        kl_div += score * np.log2(score / reco_score)

    return kl_div


ratings, movies = ml1m.load()
items_interacted_df = pd.merge(ratings, movies, on='itemid', how='inner')[['genres', 'rating']]

item_preds_df = pd.read_csv('item_based_predictions_with_genres.csv')[['genres', 'prediction']]
user_preds_df = pd.read_csv('user_based_predictions_with_genres.csv')[['genres', 'predicted_rating']]
mf_preds_df = pd.read_csv('MF_predictions_with_genres.csv')[['genres', 'score']]

interacted_distr = compute_genre_distr(items_interacted_df)

# recommended_distr = compute_genre_distr(item_preds_df)
# recommended_distr = compute_genre_distr(user_preds_df)
recommended_distr = compute_genre_distr(mf_preds_df)

interacted_distr = replace_zero_with_small_value(interacted_distr)
recommended_distr = replace_zero_with_small_value(recommended_distr)

compute_kl_divergence(interacted_distr, recommended_distr)