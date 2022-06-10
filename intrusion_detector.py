import numpy as np
import pandas as pd

import utility


class IntrusionDetector:

    def __init__(self, active_learner, classifier, initial_train_data, config_params):
        self.classifier = classifier
        self.active_learner = active_learner

        self.al_batch_size = config_params['al_batch_size']
        self.classifier_confidence_th = config_params['classifier_confidence_th']
        self.data_count_th = config_params['data_count_th']

        self.train_data_x, self.train_data_y, self.label_encoder = initial_train_data
        self.rare_data_x = np.empty((0, 78))
        self.rare_data_y = np.array([])

        self.training_class_count = np.unique(self.train_data_y).shape[0]

        self.unlabeled_data_pool = np.empty((0, 78))
        self.unlabeled_data_ids = np.array([])

    def predict(self, x, data_id):
        probabilities = self.classifier.predict(x)[0]
        class_label = np.argmax(probabilities)
        result = self.label_encoder.inverse_transform(class_label.flatten())[0]

        if not self.is_enough_confidence(probabilities):
            self.add_to_unlabeled_data_pool(x, data_id)

        if self.unlabeled_data_pool.shape[0] >= self.al_batch_size:
            self.start_active_learning_process()

        return result

    def start_active_learning_process(self):
        print("####### start active learning #####")

        # get new labeled data
        labeled_data_x, labeled_data_y = self.active_learner.get_new_labeled_data(self.classifier,
                                                                                  self.train_data_x,
                                                                                  self.train_data_y,
                                                                                  self.unlabeled_data_pool,
                                                                                  self.unlabeled_data_ids)

        # update train and rare class datasets
        self.update_data(labeled_data_x, labeled_data_y)

        # encode data
        y_train_enc, self.label_encoder = utility.encode_data(self.train_data_y)

        # reinit classifier if class count is increased
        curr_train_class_count = y_train_enc.shape[1]
        print("*** number of class: " + str(curr_train_class_count))

        if self.training_class_count != curr_train_class_count:
            self.training_class_count = curr_train_class_count
            self.classifier.reinit(self.training_class_count)
            print("!!! number of class increased")

        # retrain classifier
        self.classifier.fit(self.train_data_x, y_train_enc)
        print("Retraining completed")

        # clear data
        self.clear()

    def add_to_unlabeled_data_pool(self, x, data_id):
        self.unlabeled_data_pool = np.append(self.unlabeled_data_pool, x, axis=0)
        self.unlabeled_data_ids = np.append(self.unlabeled_data_ids, [data_id])

    def update_data(self, labeled_data_x, labeled_data_y):

        (new_train_data_x, new_train_data_y), (new_rare_data_x, new_rare_data_y) = self.split_data(labeled_data_x, labeled_data_y)

        self.rare_data_x = np.append(self.rare_data_x, new_rare_data_x, axis=0)
        self.rare_data_y = np.append(self.rare_data_y, new_rare_data_y)

        self.train_data_x = np.append(self.train_data_x, new_train_data_x, axis=0)
        self.train_data_y = np.append(self.train_data_y, new_train_data_y)

        self.transfer_data_from_rare_to_train()

    def split_data(self, labeled_data_x, labeled_data_y):
        classes = np.unique(self.train_data_y)
        mask = np.isin(labeled_data_y, classes)
        inv_mask = ~mask

        new_train_data_x = labeled_data_x[mask]
        new_train_data_y = labeled_data_y[mask]

        new_rare_data_x = labeled_data_x[inv_mask]
        new_rare_data_y = labeled_data_y[inv_mask]

        return (new_train_data_x, new_train_data_y), (new_rare_data_x, new_rare_data_y)

    def transfer_data_from_rare_to_train(self):

        classes, counts = np.unique(self.rare_data_y, return_counts=True)

        for i in range(classes.size):
            if counts[i] > self.data_count_th:

                indices = np.argwhere(self.rare_data_y == classes[i])
                new_train_data_x = self.rare_data_x[indices].reshape(-1, 78)
                new_train_data_y = self.rare_data_y[indices].reshape(1, -1)

                self.rare_data_x = np.delete(self.rare_data_x, indices, axis=0)
                self.rare_data_y = np.delete(self.rare_data_y, indices)

                self.train_data_x = np.append(self.train_data_x, new_train_data_x, axis=0)
                self.train_data_y = np.append(self.train_data_y, new_train_data_y)

    def clear(self):
        self.unlabeled_data_pool = np.empty((0, 78))
        self.unlabeled_data_ids = np.array([])

    def is_enough_confidence(self, probabilities):
        confidence = np.argmax(probabilities)
        result = False
        if confidence > self.classifier_confidence_th:
            result = True
        return result





