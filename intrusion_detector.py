import numpy as np
import sys, os, time, logging, csv, glob
import utility


class IntrusionDetector:

    def __init__(self, active_learner, classifier, initial_train_data, config_params):
        self.active_learner = active_learner
        self.classifier = classifier
        self.retrain_id = 0

        self.pool_size_th = config_params['pool_size_th']
        self.classifier_confidence_th = config_params['classifier_confidence_th']
        self.new_class_min_count_th = config_params['new_class_min_count_th']
        self.retrain_only_new_data = config_params['retrain_only_new_data']

        train_data_x, train_data_y, self.label_encoder = initial_train_data

        self.train_data_x = train_data_x.to_numpy()
        self.train_data_y = train_data_y.to_numpy()

        self.rare_data_x = np.empty((0, 78))
        self.rare_data_y = np.array([])

        self.unlabeled_data_pool = np.empty((0, 78))
        self.unlabeled_data_ids = np.array([])

        self.input_data_ids = np.array([])

        self.train_classes = np.unique(self.train_data_y)

    def predict(self, x, data_id):
        self.add_to_input_data(data_id)

        probabilities = self.classifier.predict(x)[0]
        class_index = np.argmax(probabilities)
        class_prob = probabilities[class_index]
        predicted_class_label = self.label_encoder.inverse_transform(class_index.flatten())[0]

        if not self.is_enough_confidence(probabilities):
            self.add_to_unlabeled_data_pool(x, data_id)
            # logging.info("Add to unlabeled data pool, since confidence equals to "+str(class_prob))

        data_info = None
        if self.unlabeled_data_pool.shape[0] >= self.pool_size_th:
            data_info = self.start_active_learning_process()

        return predicted_class_label, class_prob, data_info

    def start_active_learning_process(self):
        logging.info("start active learning")
        data_info = None
        self.retrain_id += 1

        input_data_info = self.extract_input_data_info()
        data_pool_info = self.extract_data_pool_info()

        # get new labeled data
        labeled_data_x, labeled_data_y = self.active_learner.get_new_labeled_data(self.classifier,
                                                                                  self.train_data_x,
                                                                                  self.train_data_y,
                                                                                  self.unlabeled_data_pool,
                                                                                  self.unlabeled_data_ids)

        labeled_data_info = self.extract_selected_data_info(labeled_data_y)

        # update train and rare class datasets
        update_success = self.update_train_data(labeled_data_x, labeled_data_y)
        train_data_info = self.extract_train_data_info()

        if update_success:
            retrain_classes = np.unique(self.train_data_y)
            is_there_new_class = self.contains_new_class(retrain_classes)

            new_class_info = self.extract_new_class_info(is_there_new_class)

            if is_there_new_class:
                curr_class_count = self.train_classes.shape[0]
                self.classifier.reinit(curr_class_count)
                logging.info("Number of class increased, current class count = " + str(curr_class_count))

            # encode data
            tmp_data_y = np.append( self.train_classes, self.train_data_y)
            y_train_enc, self.label_encoder = utility.encode_data(tmp_data_y)

            count = self.train_classes.shape[0]
            y_train_enc = y_train_enc[count:]

            # retrain classifier
            self.classifier.fit(self.train_data_x, y_train_enc)
            logging.info("Retraining-" + str(self.retrain_id) + " is completed")

            data_info = (input_data_info, data_pool_info, labeled_data_info, train_data_info, new_class_info)

        else:
            self.retrain_id -= 1

        # clear data
        self.clear()

        return data_info

    def add_to_unlabeled_data_pool(self, x, data_id):
        self.unlabeled_data_pool = np.append(self.unlabeled_data_pool, x, axis=0)
        self.unlabeled_data_ids = np.append(self.unlabeled_data_ids, [data_id])

    def add_to_input_data(self, data_id):
        self.input_data_ids = np.append(self.input_data_ids, [data_id])

    def update_train_data(self, labeled_data_x, labeled_data_y):
        if self.retrain_only_new_data:
            self.train_data_x = np.empty((0, 78))
            self.train_data_y = np.array([])

        update_success = False
        train_data_count = np.size(self.train_data_y)

        (new_train_data_x, new_train_data_y), (new_rare_data_x, new_rare_data_y) = self.split_rare_and_train_data(labeled_data_x, labeled_data_y)

        self.rare_data_x = np.append(self.rare_data_x, new_rare_data_x, axis=0)
        self.rare_data_y = np.append(self.rare_data_y, new_rare_data_y)

        self.train_data_x = np.append(self.train_data_x, new_train_data_x, axis=0)
        self.train_data_y = np.append(self.train_data_y, new_train_data_y)

        self.transfer_data_from_rare_to_train()

        curr_train_data_count = np.size(self.train_data_y)

        if curr_train_data_count > train_data_count:
            update_success = True
        return update_success

    def split_rare_and_train_data(self, labeled_data_x, labeled_data_y):
        mask = np.isin(labeled_data_y, self.train_classes)
        inv_mask = ~mask

        new_train_data_x = labeled_data_x[mask]
        new_train_data_y = labeled_data_y[mask]

        new_rare_data_x = labeled_data_x[inv_mask]
        new_rare_data_y = labeled_data_y[inv_mask]

        return (new_train_data_x, new_train_data_y), (new_rare_data_x, new_rare_data_y)

    def transfer_data_from_rare_to_train(self):
        classes, counts = np.unique(self.rare_data_y, return_counts=True)

        for i in range(classes.size):
            if counts[i] > self.new_class_min_count_th:

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
        self.input_data_ids = np.array([])

    def is_enough_confidence(self, probabilities):
        index = np.argmax(probabilities)
        confidence = probabilities[index]
        result = False
        if confidence > self.classifier_confidence_th:
            result = True
        return result

    def update_pool_size_th(self):
        pass

    def extract_input_data_info(self):
        labels = self.active_learner.ask_simulated_oracle(self.input_data_ids)
        classes, counts = np.unique(labels, return_counts=True)
        info = self.get_info_table(classes, counts, "input_data")
        return info

    def extract_data_pool_info(self):
        labels = self.active_learner.ask_simulated_oracle(self.unlabeled_data_ids)
        classes, counts = np.unique(labels, return_counts=True)
        info = self.get_info_table(classes, counts, "unlabeled_data_pool")
        return info

    def extract_selected_data_info(self, labeled_data_y):
        classes, counts = np.unique(labeled_data_y, return_counts=True)
        info = self.get_info_table(classes, counts, "selected_data")
        return info

    def extract_train_data_info(self):
        classes, counts = np.unique(self.train_data_y, return_counts=True)
        info = self.get_info_table(classes, counts, "train_data")
        return info

    def extract_new_class_info(self, is_there_new_class):
        name = "train_data-" + str(self.retrain_id)
        val = 1 if is_there_new_class else 0
        info = {'name': name, 'new_class': val}
        return info

    def get_info_table(self, classes, counts, name):
        complete_class_list = self.active_learner.get_classes()
        name = name+"-"+str(self.retrain_id)
        info = {'name': name}
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

    def contains_new_class(self, retrain_classes):
        result = False
        class_count = self.train_classes.shape[0]

        tmp = np.append(self.train_classes, retrain_classes)
        self.train_classes = np.unique(tmp)

        new_class_count = self.train_classes.shape[0]

        if new_class_count > class_count:
            result = True
        return result
