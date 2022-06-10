import numpy as np

import utility


class IntrusionDetector:

    def __init__(self, anomaly_detector, classifier, active_learner, initial_train_data, batch_size, confidence_threshold):

        self.anomaly_detector = anomaly_detector
        self.classifier = classifier
        self.active_learner = active_learner

        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold

        self.current_data_count = 0

        self.train_data_x, self.train_data_y, self.label_encoder = initial_train_data
        self.num_of_class = np.shape(np.unique(self.train_data_y))[0]

        self.unlabeled_data_pool = np.empty((0, 78))
        self.unlabeled_data_ids = np.array([])

    def predict(self, x, data_id=None):
        result = None

        # step 1: Is it attack or normal data?
        label = self.anomaly_detector.predict(x)[0]

        if label == 'BENIGN':
            result = label
        else:
            self.current_data_count += 1
            # step 2: Which kind of attack it is?
            probabilities = self.classifier.predict(x)[0]
            class_label = np.argmax(probabilities)

            result = self.label_encoder.inverse_transform(class_label.flatten())[0]

            if not self.is_enough_confidence(probabilities):
                self.add_to_unlabeled_data_pool(x, data_id)

            if self.current_data_count >= self.batch_size and self.unlabeled_data_pool.shape[0] != 0:
                if self.active_learner is not None:
                    self.start_active_learning_process()
                else:
                    self.clear()

        return result

    def start_active_learning_process(self):
        print("####### start active learning #####")
        labeled_data_x, labeled_data_y = self.active_learner.get_new_labeled_data(self.classifier,
                                                                                  self.train_data_x,
                                                                                  self.train_data_y,
                                                                                  self.unlabeled_data_pool,
                                                                                  self.unlabeled_data_ids)
        self.update_train_data(labeled_data_x, labeled_data_y)
        y_train_enc, self.label_encoder = utility.encode_data(self.train_data_y)
        num_of_class = y_train_enc.shape[1]
        print("*** number of class: " + str(num_of_class))

        if self.num_of_class != num_of_class:
            self.num_of_class = num_of_class
            self.classifier.reinit(self.num_of_class)
            print("!!! number of class increased")

        self.classifier.fit(self.train_data_x, y_train_enc)
        print("Retraining completed")
        self.clear()

    def add_to_unlabeled_data_pool(self, x, data_id=None):
        self.unlabeled_data_pool = np.append(self.unlabeled_data_pool, x, axis=0)
        if data_id is not None:
            self.unlabeled_data_ids = np.append(self.unlabeled_data_ids, [data_id])

    def update_train_data(self, labeled_data_x, labeled_data_y):
        # remove 'benign' data from labeled data
        indices = np.argwhere(labeled_data_y == 'BENIGN')
        labeled_data_x = np.delete(labeled_data_x, indices, axis=0)
        labeled_data_y = np.delete(labeled_data_y, indices)

        self.train_data_x = np.append(self.train_data_x, labeled_data_x, axis=0)
        self.train_data_y = np.append(self.train_data_y, [labeled_data_y])

    def clear(self):
        self.unlabeled_data_pool = np.empty((0, 78))
        self.unlabeled_data_ids = np.array([])
        self.current_data_count = 0

    def is_enough_confidence(self, probabilities):
        confidence = np.argmax(probabilities)
        result = False
        if confidence > self.confidence_threshold:
            result = True
        return result





