# Machine Learning By Example

Hands-on machine learning projects built to develop practical, end-to-end understanding of core ML techniques.

In this repository each project is implemented to understand both the theory and the engineering behind model training, evaluation, and real-world tradeoffs.

## Projects

### Naive Bayes

Location: `naive-bayes/classifiers/`

- `naive-bayes-from-scratch.py`: Bernoulli-style Naive Bayes implemented manually on a small movie preference example.
- `naive-bayes-scikit-toy.py`: The same toy classification problem solved with `scikit-learn`'s `BernoulliNB`.
- `movie-recommender.py`: A MovieLens-based recommendation experiment using `MultinomialNB`, with accuracy, classification reporting, and ROC/AUC evaluation.
- `heart-disease-classifier.py`: Binary heart disease classification using the UCI Heart Disease dataset, including stratified k-fold validation for parameter selection.

Topics covered:
- prior, likelihood, and posterior calculation
- feature preparation and label thresholding
- train/test splitting
- model evaluation with classification metrics and ROC/AUC
- basic hyperparameter selection

### Deep Learning

Location: `deep-learning/`

#### MNIST From Scratch

Location: `deep-learning/mnist-from-scratch/`

- `code/read_mnist.py`: Reads MNIST IDX image and label files into NumPy arrays.
- `code/nn_from_scratch.py`: NumPy implementation of a fully connected neural network with ReLU activations, softmax output, backpropagation, and mini-batch SGD.

What it demonstrates:
- manual weight and bias initialization
- forward propagation through dense layers
- softmax plus cross-entropy gradient flow
- backpropagation and parameter updates
- training and testing on MNIST digit data

#### PyTorch Notebook

Location: `deep-learning/pytorch/`

- `hotel-cancellations.ipynb`: Notebook project for predicting hotel booking cancellations using neural networks for binary and multiclass classification with PyTorch.

## Data

- `datasets/ml-1m/`: MovieLens files used for the movie recommendation work.
- `datasets/hotel_bookings.csv`: Hotel bookings dataset used by the PyTorch notebook.
- `deep-learning/mnist-from-scratch/dataset/`: Local MNIST IDX files for the scratch neural network project.

## Tech Stack

- Python
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- PyTorch
- Jupyter Notebook
- `ucimlrepo`

## Repository Structure

```text
apps/
├── datasets/
│   ├── hotel_bookings.csv
│   └── ml-1m/
├── deep-learning/
│   ├── mnist-from-scratch/
│   │   ├── code/
│   │   └── dataset/
│   └── pytorch/
│       └── hotel-cancellations.ipynb
└── naive-bayes/
    └── classifiers/
        ├── heart-disease-classifier.py
        ├── movie-recommender.py
        ├── naive-bayes-from-scratch.py
        └── naive-bayes-scikit-toy.py
```

## Running Projects

Run scripts from the `apps` directory:

```bash
python naive-bayes/classifiers/naive-bayes-from-scratch.py
python naive-bayes/classifiers/naive-bayes-scikit-toy.py
python naive-bayes/classifiers/heart-disease-classifier.py
python deep-learning/mnist-from-scratch/code/nn_from_scratch.py
```

Open the notebook with Jupyter for the PyTorch project:

```bash
jupyter notebook deep-learning/pytorch/hotel-cancellations.ipynb
```

## Notes

- The heart disease classifier fetches data through `ucimlrepo`.
- A few scripts use project-relative paths, so dataset locations may need to match the structure shown above.
