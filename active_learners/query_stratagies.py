import numpy as np
import copy


# returns score for each data in the unlabeled dataset according query strategy
from scipy.stats import entropy


def uncertainty_sampling(classifier, unlabeled_data):
    probabilities = classifier.predict(unlabeled_data)
    scores = 1 - np.amax(probabilities, axis=1)
    return scores


def uncertainty_margin_sampling(classifier, unlabeled_data):
    probabilities = classifier.predict(unlabeled_data)
    part = np.partition(-probabilities, 1, axis=1)
    scores = - part[:, 0] + part[:, 1]
    return scores


def uncertainty_entropy_sampling(classifier, unlabeled_data):
    probabilities = classifier.predict(unlabeled_data)
    scores = np.transpose(entropy(np.transpose(probabilities)))
    return scores


def expected_model_change_sampling(classifier, unlabeled_data, labeled_data, labels):
    probabilities = classifier.predict(unlabeled_data)
    predictions = classifier.predict(unlabeled_data)

    scores = []
    for i, x in enumerate(unlabeled_data):
        sample_data = x
        predicted_label = predictions[i]

        X = np.append(labeled_data, [sample_data])
        y = np.append(labels, predicted_label)

        m = copy.deepcopy(classifier)
        m = m.fit(X, y)

        new_probabilities = m.predict(unlabeled_data)
        new_predictions = m.predict(unlabeled_data)

        label_change = np.abs(new_predictions - predictions)
        prediction_change = np.abs(
            np.apply_along_axis(np.max, 1, new_probabilities) - np.apply_along_axis(np.max, 1, probabilities))

        sum_change = np.array([a * b for a, b in zip(label_change, prediction_change)])

        score = np.sum(sum_change)
        scores.append(score)

    return scores


def expected_error_reduction_sampling(classifier, unlabeled_data, labeled_data, labels):
    classes = np.unique(labels)
    n_classes = len(classes)

    probabilities = classifier.predict_proba(unlabeled_data)

    scores = []
    for i, sample_data in enumerate(unlabeled_data):
        tmp_train_data = np.append(labeled_data, [sample_data])

        score = []
        for yi in range(n_classes):
            tmp_labels = np.append(labels, yi)
            
            m = copy.deepcopy(classifier)

            m.fit(tmp_train_data, tmp_labels)
            
            p = m.predict_proba(unlabeled_data)
            # score.append(probabilities[i, yi] * np.sum(1 - np.max(p, axis=1)))  # 0/1 loss
            score.append(probabilities[i, yi] * -np.sum(p * np.log(p)))  # log loss
            
        scores.append(np.sum(score))

    return scores
