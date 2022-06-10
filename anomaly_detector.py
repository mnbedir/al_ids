from sklearn.neighbors import KNeighborsClassifier


class AnomalyDetector:

    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=5)

    def fit(self, train_data_x, train_data_y):
        self.knn.fit(train_data_x, train_data_y)

    def predict(self, data):
        return self.knn.predict(data)
