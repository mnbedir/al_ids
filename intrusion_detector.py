import numpy as np


class IntrusionDetector:

    def __init__(self, anomaly_detector, classifier, active_learner, initial_train_data, batch_size, confidence_threshold):

        self.anomaly_detector = anomaly_detector
        self.classifier = classifier
        self.active_learner = active_learner

        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold

        self.current_data_count = 0

        self.train_data_x, self.train_data_y, self.label_encoder = initial_train_data
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

            result = self.label_encoder.inverse_transform(class_label.flatten())

            if not self.is_enough_confidence(probabilities):
                self.add_to_unlabeled_data_pool(x, data_id)

            if self.current_data_count == self.batch_size and self.unlabeled_data_pool.shape[0] != 0:
                if self.active_learner is not None:
                    self.start_active_learning_process()
                else:
                    self.clear()

        return result

    def start_active_learning_process(self):
        labeled_data_x, labeled_data_y = self.active_learner.get_new_labeled_data(self.classifier,
                                                                                  self.train_data_x,
                                                                                  self.train_data_y,
                                                                                  self.unlabeled_data_pool,
                                                                                  self.unlabeled_data_ids)
        self.update_train_data(labeled_data_x, labeled_data_y)
        self.classifier.fit(self.train_data_x, self.train_data_y)
        self.clear()

    def add_to_unlabeled_data_pool(self, x, data_id=None):
        self.unlabeled_data_pool = np.append(self.unlabeled_data_pool, x, axis=0)
        if data_id is not None:
            self.unlabeled_data_ids = np.append(self.unlabeled_data_ids, [data_id])

    def update_train_data(self, labeled_data_x, labeled_data_y):
        self.train_data_x = np.append(self.train_data_x, labeled_data_x)
        self.train_data_y = np.append(self.train_data_y, labeled_data_y)

    def clear(self):
        self.unlabeled_data_pool = np.delete(self.unlabeled_data_pool)
        self.unlabeled_data_ids = np.delete(self.unlabeled_data_ids)
        self.current_data_count = 0

    def is_enough_confidence(self, probabilities):
        confidence = np.argmax(probabilities)
        result = False
        if confidence > self.confidence_threshold:
            result = True
        return result





