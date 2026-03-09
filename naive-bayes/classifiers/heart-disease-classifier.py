import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# -----
# Prepare data

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# initialising new variable because I was unable to modify `X.values`.
X_no_nan = np.nan_to_num(X.values, nan=0)

# set heart_disease threshold for binary classification
# threshold of 1 means model will predict if heart disease is present or not
threshold = 1
y.values[y.values >= threshold] = 1

# variable information
# uncomment if you would like to see what each variable means
# print(heart_disease.variables)

# -----
# k-fold cross-validation

k = 5  # number of folds
k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# model parameters to optimise
smoothing_factor_options = [1, 2, 3, 4, 5, 6]
fit_prior_options = [True, False]
auc_record = {}

# calculate the average area under the roc auc curve for each smoothing factor and fit prior
for train_indices, test_indices in k_fold.split(X_no_nan, y.values.ravel()):
    X_train_k, X_test_k = X_no_nan[train_indices], X_no_nan[test_indices]
    Y_train_k, Y_test_k = y.values[train_indices], y.values[test_indices]
    for alpha in smoothing_factor_options:
        if alpha not in auc_record:
            auc_record[alpha] = {}
        for fit_prior in fit_prior_options:
            clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            clf.fit(X_train_k, Y_train_k.ravel())
            prediction_prob = clf.predict_proba(X_test_k)
            pos_prob = prediction_prob[:, 1]
            auc = roc_auc_score(Y_test_k, pos_prob)
            auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)
# Uncomment to find the best performing model parameters
# for smoothing, smoothing_record in auc_record.items():
#     for fit_prior, auc in smoothing_record.items():
#         print(f"    {smoothing}         {fit_prior}         {auc/k:.5f}")

# -----
# Train NB model

X_train, X_test, y_train, y_test = train_test_split(
    X_no_nan, y.values, test_size=0.20, random_state=42, stratify=y.values
)

# use the best performing smoothing factor and fit prior option
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, y_train.ravel())

# -----
# Test NB model
prediction_prob = clf.predict_proba(X_test)
prediction = clf.predict(X_test)

accuracy = clf.score(X_test, y_test)
print(f"The accuracy is: {accuracy*100:.1f}%")

# -----
# Evaluate NB model
report = classification_report(y_test, prediction)
print(report)
