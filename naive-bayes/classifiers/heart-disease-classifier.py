import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

X_train, X_test, y_train, y_test = train_test_split(X_no_nan, y.values, test_size=0.2)

# -----
# Train NB model

clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, y_train)

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
