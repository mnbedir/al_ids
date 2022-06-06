import numpy as np


def ask_oracle(unlabeled_data):
    labels = np.array([])
    return labels


def ask_simulated_oracle(selected_indices, dataset_labels):
    labels = dataset_labels[selected_indices]
    return labels


def update_labeled_train_dataset(new_labeled_data, labeled_data, new_labels, labels):
    labeled_data = np.append(labeled_data, new_labeled_data)
    labels = np.append(labels, new_labels)


def updated_unlabeled_dataset(selected_indices, unlabeled_data):
    unlabeled_data = np.delete(unlabeled_data, selected_indices)


def retrain_classifier(classifier, updated_labeled_data, updated_labels):
    classifier.fit(updated_labeled_data, updated_labels)



