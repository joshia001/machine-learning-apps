import numpy as np

# TOY SCENARIO - SUPERVISED LEARNING BY HAND 
# KNOWLEDGE WILL FEED INTO A LARGER APPLICATION USING ACTUAL DATASETS
#--------------------------------------------------------------------
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

def get_label_indices(labels):
    """
    Group samples based on their labels and return indices
    @param labels: list of labels
    @return: dict, {class1: [indices], class2: [indices]}
    """
    from collections import defaultdict
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices

label_indices = get_label_indices(y_train)
print('label_indices:\n', label_indices)

def get_prior(label_indices):
    """
    Compute prior probabilities from class (label) distribution
    @param label_indices: dict, {class1: [indices], class2: [indices]}
    @return: dict, {class1: prior1, class2: prior2}
    """
    prior =  {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior

prior = get_prior(label_indices)
print('Prior:', prior)

def get_likelihood(features, label_indices, smoothing=0):
    """
    Compute likelihood based on training samples
    @param features: matrix of features
    @param label_indices: grouped sample indices by class
    @param smoothing: integer, additive smoothing parameter
    @return: dictionary, with class as key, corresponding
             conditional probability P(feature|class) vector
             as value    
    """
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)
    return likelihood

smoothing = 1
likelihood = get_likelihood(x_train, label_indices, smoothing)
print('Likelihood: \n', likelihood)

def get_posterior(X, prior, likelihood):
    """
    Compute posterior of testing samples, based on prior and
    likelihood
    @param x: testing samples
    @param prior: dictionary, with class label as key,
                  corresponding prior as the value
    @param likelihood: dictionary, with class label as key,
                       corresponding conditional probability
                       vector as value
    @return: dictionary, with class label as key, corresponding
             posterior as value
    """
    posteriors = []
    for x in X:
        # posterior is proportional to prior * likelihood
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                if bool_value:
                    posterior[label] *= likelihood_label[index]
                else: 
                    posterior[label] *= (1-likelihood_label[index])
            # normalise so everything sums up to 1
            sum_posterior = sum(posterior.values())
            for label in posterior:
                if posterior[label] == float('inf'):
                    posterior[label] = 1.0
                else:
                    posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors

posterior = get_posterior(x_test, prior, likelihood)
print('Posterior:\n', posterior)
