from sklearn.naive_bayes import BernoulliNB
import numpy as np

# Scenario: Given we know if users like movies A, B or C, we want to predict if they will like movie D.
# The training dataset is labelled (supervised learning).

# Features matrix used as training set - in this case depicting whether a person liked movie A, B, C or not.
# E.g Person 0 didn't like movie A but liked movies B and C.  
x_train = np.array([[0, 1, 1],
                   [0, 0, 1],
                   [0, 0, 0],
                   [1, 1, 0]])

# Labels/classes per dataset sample
# E.g Person 0 liked movie D
y_train = ['Y', 'N', 'Y', 'Y']

# Testing dataset
x_test = np.array([[1, 1, 0]])

classifier = BernoulliNB(alpha=1.0, fit_prior=True)

classifier.fit(x_train, y_train)

pred_prob = classifier.predict_proba(x_test)
print('[scikit-learn] Predicted probabilities:\n', pred_prob)

prediction = classifier.predict(x_test)
print('[scikit-learn] Prediction:', prediction)