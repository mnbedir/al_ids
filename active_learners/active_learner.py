import numpy as np

from active_learners.query_stratagies import *
from active_learners.selection_strategies import *


class ActiveLearner:
    def __init__(self, query_strategy, selection_strategy, selection_param, labels=None):
        self.query_strategy = query_strategy
        self.selection_strategy = selection_strategy
        self.selection_param = selection_param
        self.labels = labels

    def get_new_labeled_data(self, classifier, train_data_x, train_data_y, unlabeled_data_pool, unlabeled_data_ids):
        scores = self.query(classifier, train_data_x, train_data_y, unlabeled_data_pool)
        selected_indices = self.select(scores)

        selected_unlabeled_data, selected_data_ids = self.get_selected_data_info(selected_indices,
                                                                                 unlabeled_data_pool,
                                                                                 unlabeled_data_ids)
        selected_data_labels = self.ask(selected_unlabeled_data, selected_data_ids)

        return selected_unlabeled_data, selected_data_labels

    def query(self, classifier, train_data_x, train_data_y, unlabeled_data_pool):
        if self.query_strategy == 'expected_error_reduction':
            scores = expected_error_reduction_sampling(classifier, unlabeled_data_pool, train_data_x, train_data_y)
        elif self.query_strategy == 'expected_model_change':
            scores = expected_model_change_sampling(classifier, unlabeled_data_pool, train_data_x, train_data_y)
        else:
            scores = uncertainty_sampling(classifier, unlabeled_data_pool)
        return scores

    def select(self, scores):
        if self.selection_strategy == 'max_n':
            selected_indices = select_max_n_sample(scores, self.selection_param)
        elif self.selection_strategy == 'min_n':
            selected_indices = select_min_n_sample(scores, self.selection_param)
        elif self.selection_strategy == 'gt_th':
            selected_indices = select_sample_score_greater_than(scores, self.selection_param)
        else:
            selected_indices = select_sample_score_less_than(scores, self.selection_param)
        return selected_indices

    def get_selected_data_info(self, selected_indices, unlabeled_data_pool, unlabeled_data_ids):
        selected_unlabeled_data = unlabeled_data_pool[selected_indices]
        if len(unlabeled_data_ids) != 0:
            selected_data_ids = unlabeled_data_ids[selected_indices]
        else:
            selected_data_ids = None

        return selected_unlabeled_data, selected_data_ids

    def ask_simulated_oracle(self, selected_data_ids):
        selected_data_labels = self.labels[selected_data_ids]
        return selected_data_labels

    def ask_oracle(self, selected_unlabeled_data):
        # ask oracle via gui
        selected_data_labels = np.array([])
        return selected_data_labels

    def ask(self, selected_unlabeled_data, selected_data_ids):
        if self.labels is None:
            selected_data_labels = self.ask_simulated_oracle(selected_data_ids)
        else:
            selected_data_labels = self.ask_oracle(selected_unlabeled_data)
        return selected_data_labels



