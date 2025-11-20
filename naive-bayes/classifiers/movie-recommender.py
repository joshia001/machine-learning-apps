import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

#-----
# Prepare data

ROOT = Path(__file__).resolve().parents[1]

data_path = ROOT / 'datasets'/ 'ml-1m' / 'ratings.dat'
df = pd.read_csv(data_path, header=None, sep='::', engine='python')
df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
# print('Dataframe: \n', df)
n_users = df['user_id'].nunique()
n_movies = df['movie_id'].nunique()
# print(f'# Users: {n_users}')
# print(f'# Movies: {n_movies}')

def load_user_rating_data(df, n_users, n_movies):
    data = np.empty([n_users, n_movies])
    movie_id_mapping = {}
    for user_id, movie_id, rating in zip(df['user_id'], df['movie_id'],
                                         df['rating']):
        # start user_id from 0 to match matrix indices
        user_id = int(user_id) - 1
        if movie_id not in movie_id_mapping:
            # maps movie_id to data column
            movie_id_mapping[movie_id] = len(movie_id_mapping)
        data[user_id, movie_id_mapping[movie_id]] = rating    
    return data, movie_id_mapping
data, movie_id_mapping = load_user_rating_data(df, n_users, n_movies)

# analyse data distribution to identify any class imbalances
values, counts = np.unique(data, return_counts=True)
# for value, count in zip(values, counts):
    # print(f'Number of rating {value}: {count}')

# most ratings are unknown. not all users have rated all movies. 
# take movie with the most known ratings as our target movie.
# look for rating counts for each movie:
# print(df['movie_id'].value_counts())
# target movie is ID and ratings of other movies are treated as features
target_movie_id = 2858
X_raw = np.delete(data, movie_id_mapping[target_movie_id], axis=1)
Y_raw = data[:, movie_id_mapping[target_movie_id]]
X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]
# print('Shape of X:', X.shape)
# print('Shape of Y:', Y.shape)

recommend = 3 # threshold (ratings > 3 mean movie is liked)
Y[Y <= recommend] = 0
Y[Y > recommend] = 1
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
# print(f'{n_pos} positive samples and {n_neg} negative samples.')

# split data into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
    test_size=0.2, random_state=42)

#-----
# Train NB model

clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)

#-----
# Test NB model
prediction_prob = clf.predict_proba(X_test)
prediction = clf.predict(X_test)

accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is: {accuracy*100:.1f}%')

#-----
# Evaluate model

# confusion matrix
# conf_mtx = confusion_matrix(Y_test, prediction, labels=[0,1])

# precision
# precision = conf_mtx[1][1] / (conf_mtx[1][1]+conf_mtx[0][1])
# precision = precision_score(Y_test, prediction, pos_label=1)
# 0.90

# recall
# recall = conf_mtx[1][1] / (conf_mtx[1][1]+conf_mtx[1][0])
# recall = recall_score(Y_test, prediction, pos_label=1)
# 0.74

# f1 score
# f1 = f1_score(Y_test, prediction, pos_label=1)
# 0.82

# classification report will compute all of the above evaluation metrics
report = classification_report(Y_test, prediction)
print(report)

# area under curve (AUC) of receiver operating characteristic (ROC)
pos_prob = prediction_prob[:, 1]
thresholds = np.arange(0.0, 1.1, 0.05)
true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
for pred, y in zip(pos_prob, Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:
            # if truth and prediction are both 1
            if y == 1:
                true_pos[i] += 1
            # if truth is 0 while prediction is 1
            else:
                false_pos[i] += 1
        else:
            break

n_pos_test = (Y_test == 1).sum()
n_neg_test = (Y_test == 0).sum()
true_pos_rate = [tp / n_pos_test for tp in true_pos]
false_pos_rate = [fp / n_neg_test for fp in false_pos]
# compute AUC of model
roc_auc = roc_auc_score(Y_test, pos_prob)
print('AUC = ', roc_auc)
# plot ROC
plt.figure()
lw = 2
plt.plot(false_pos_rate, true_pos_rate,
         color='darkorange', lw=lw, label='model ROC')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',
         label='random guessing')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()
