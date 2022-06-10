import numpy as np

# return the indices selected according to score and selection strategy


def select_max_n_sample(scores, n):
    size = np.shape(scores)[0]
    if size < n:
        n = size
    selected_indices = np.argsort(scores)[-n:]
    return selected_indices


def select_min_n_sample(scores, n):
    size = np.shape(scores)[0]
    if size < n:
        n = size
    selected_indices = np.argsort(scores)[:n]
    return selected_indices


def select_sample_score_greater_than(scores, threshold):
    pass


def select_sample_score_less_than(scores, threshold):
    pass
