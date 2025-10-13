import numpy as np

x_train = np.array([[0, 1, 1],
                   [0, 0, 1],
                   [0, 0, 0],
                   [1, 1, 0]])

y_train = ['Y', 'N', 'Y', 'Y']
x_test = np.array([1, 1, 0])

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

