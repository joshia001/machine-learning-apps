# Machine Learning By Example

Hands-on machine learning projects built to develop practical, end-to-end understanding of core ML techniques.

In this repository each project is implemented to understand both the *theory* and the *engineering* behind model training, evaluation, and real-world tradeoffs.

## Why This Repository Exists

The goal is to build broad, practical mastery across machine learning domains by implementing models from:
- first principles (to understand internals), and
- production-grade libraries (to compare against standard tooling).

## Current Work

### 1) Naive Bayes Classifiers
Location: `naive-bayes/classifiers/`

Implemented classifiers include:
- `naive-bayes-from-scratch.py`: Manual implementation of a Bernoulli-style Naive Bayes workflow (prior, likelihood, posterior) on a toy movie-preference problem.
- `naive-bayes-scikit-toy.py`: Equivalent toy example using `scikit-learn`'s `BernoulliNB` for comparison.
- `movie-recommender.py`: Multinomial Naive Bayes movie recommendation experiment using MovieLens ratings data, with ROC/AUC analysis and classification reporting.
- `heart-disease-classifier.py`: Multinomial Naive Bayes applied to UCI Heart Disease data, including k-fold cross-validation for hyperparameter selection.

Key skills demonstrated:
- data preparation and feature shaping,
- class thresholding and binary conversion,
- model training/testing pipelines,
- evaluation with precision/recall/F1, ROC, and AUC,
- hyperparameter exploration via cross-validation.

### 2) Neural Network From Scratch (MNIST)
Location: `neural-net/mnist/`

Implemented components include:
- `code/read_mnist.py`: Binary IDX file loader for raw MNIST images/labels.
- `code/nn_from_scratch.py`: Fully connected neural network implementation in NumPy.
- `code/show_mnist.py`: Dataset visualization and model experimentation script.

Neural network features:
- configurable dense architecture (default: 784 -> 64 -> 64 -> 10),
- ReLU hidden activations,
- softmax output,
- cross-entropy gradient with backpropagation,
- mini-batch SGD training loop,
- accuracy tracking during training/testing.

## Tech Stack

- Python
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- UCI ML Repository API (`ucimlrepo`)

## Repository Structure

```text
apps/
├── naive-bayes/
│   └── classifiers/
│       ├── heart-disease-classifier.py
│       ├── movie-recommender.py
│       ├── naive-bayes-from-scratch.py
│       └── naive-bayes-scikit-toy.py
└── neural-net/
    └── mnist/
        ├── code/
        │   ├── nn_from_scratch.py
        │   ├── read_mnist.py
        │   └── show_mnist.py
        └── dataset/
```

## Running Projects

From the `apps` directory, run scripts directly:

```bash
python naive-bayes/classifiers/heart-disease-classifier.py
python naive-bayes/classifiers/movie-recommender.py
python neural-net/mnist/code/nn_from_scratch.py
```

Note:
- Some scripts expect local datasets to already exist in the referenced paths.
- The heart disease project fetches data through `ucimlrepo`.

More projects are currently underway.
