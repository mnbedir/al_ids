import numpy as np
import sys, os, time, logging, csv, glob
import utility


class IntrusionDetector:

    def __init__(self, active_learner, classifier, initial_train_data, config_params):
        self.active_learner = active_learner
        self.classifier = classifier
        self.input_counter = 0
        self.retrain_id = 0

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
        self.input_counter += 1

        probabilities = self.classifier.predict(x)[0]
        class_index = np.argmax(probabilities)
        class_prob = probabilities[class_index]
        predicted_class_label = self.label_encoder.inverse_transform(class_index.flatten())[0]
        data_info = None

        if not self.is_enough_confidence(probabilities):
            self.add_to_unlabeled_data_pool(x, data_id)
            # logging.info("Add to unlabeled data pool, since confidence equals to "+str(class_prob))

        if self.input_counter >= self.al_batch_size and self.unlabeled_data_pool.shape[0] > 0:
            data_info = self.start_active_learning_process()

        return predicted_class_label, class_prob, data_info

    def start_active_learning_process(self):
        logging.info("start active learning")
        data_info = None

        data_pool_info = self.extract_data_pool_info()

        # get new labeled data
        labeled_data_x, labeled_data_y = self.active_learner.get_new_labeled_data(self.classifier,
                                                                                  self.train_data_x,
                                                                                  self.train_data_y,
                                                                                  self.unlabeled_data_pool,
                                                                                  self.unlabeled_data_ids)

        labeled_data_info = self.extract_selected_data_info(labeled_data_y)

        # update train and rare class datasets
        update_success = self.update_data(labeled_data_x, labeled_data_y)
        train_data_info = self.extract_train_data_info()

        if update_success:
            # encode data
            y_train_enc, self.label_encoder = utility.encode_data(self.train_data_y)

            # reinit classifier if class count is increased
            curr_train_class_count = y_train_enc.shape[1]

            if self.training_class_count != curr_train_class_count:
                self.training_class_count = curr_train_class_count
                self.classifier.reinit(self.training_class_count)
                logging.info("Number of class increased, new class count = " + str(curr_train_class_count))

            self.retrain_id += 1
            # retrain classifier
            self.classifier.fit(self.train_data_x, y_train_enc)
            logging.info("Retraining-" + str(self.retrain_id) + " is completed")
            data_info = (data_pool_info, labeled_data_info, train_data_info)

        # clear data
        self.clear()

        return data_info

    def add_to_unlabeled_data_pool(self, x, data_id):
        self.unlabeled_data_pool = np.append(self.unlabeled_data_pool, x, axis=0)
        self.unlabeled_data_ids = np.append(self.unlabeled_data_ids, [data_id])

    def update_data(self, labeled_data_x, labeled_data_y):
        update_success = False
        train_data_count = np.size(self.train_data_y)

        (new_train_data_x, new_train_data_y), (new_rare_data_x, new_rare_data_y) = self.split_data(labeled_data_x, labeled_data_y)

        self.rare_data_x = np.append(self.rare_data_x, new_rare_data_x, axis=0)
        self.rare_data_y = np.append(self.rare_data_y, new_rare_data_y)

        self.train_data_x = np.append(self.train_data_x, new_train_data_x, axis=0)
        self.train_data_y = np.append(self.train_data_y, new_train_data_y)

        self.transfer_data_from_rare_to_train()

        curr_train_data_count = np.size(self.train_data_y)

        if curr_train_data_count > train_data_count:
            update_success = True
        return update_success

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
        self.input_counter = 0

    def is_enough_confidence(self, probabilities):
        index = np.argmax(probabilities)
        confidence = probabilities[index]
        result = False
        if confidence > self.classifier_confidence_th:
            result = True
        return result

    def extract_data_pool_info(self):
        labels = self.active_learner.ask_simulated_oracle(self.unlabeled_data_ids)
        classes, counts = np.unique(labels, return_counts=True)
        info = self.get_info_table(classes, counts)
        return info

    def extract_selected_data_info(self, labeled_data_y):
        classes, counts = np.unique(labeled_data_y, return_counts=True)
        info = self.get_info_table(classes, counts)
        return info

    def extract_train_data_info(self):
        classes, counts = np.unique(self.train_data_y, return_counts=True)
        info = self.get_info_table(classes, counts)
        return info

    def get_info_table(self, classes, counts):
        complete_class_list = self.active_learner.get_classes()
        info = {}
        for i in range(complete_class_list.size):
            curr_class = complete_class_list[i]
            indices = np.where(classes == curr_class)
            if np.size(indices) != 0:
                info[curr_class] = counts[indices[0][0]]
            else:
                info[curr_class] = 0

        info['total'] = np.sum(counts)
        return info

    def get_attack_classifier(self):
        return self.classifier

    def get_label_encoder(self):
        return self.label_encoder


