import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        """
        Saves training data
        """

        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=1):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loop(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        distances = np.zeros((X.shape[0], self.train_X.shape[0]))
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                distances[i][j] = np.sum(np.abs(X[[i]] - self.train_X[[j]]))
        return distances


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        distances = np.zeros((X.shape[0], self.train_X.shape[0]))
        for i in range(distances.shape[0]):
            distances[i] = np.sum(np.abs(X[i] - self.train_X), axis=1)
        return distances


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        distances = np.zeros((X.shape[0], self.train_X.shape[0]))
        distances += np.sum(np.abs(self.train_X[:, None] - X[None, :]), axis=2).T
        return distances


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        prediction, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_test = distances.shape[0]
        prediction = np.zeros(n_test)
        
        for i in range(n_test):
            # finding indexes of the nearest k neighbours
            nearest_k_ind = np.argsort(distances[i])[:self.k]
            # extracting the nearest k neighbours from train_y
            nearest_k = np.array([self.train_y[ind] for ind in nearest_k_ind])
            # counting the number of occurrences of each value in nearest_k
            # the input array for np.bincount needs to be of integer dtype, otherwise a TypeError is raised
            counts = np.bincount(nearest_k.astype('int'))
            prediction[i] = np.argmax(counts)
            prediction = prediction.astype('int64').astype('str')
        return prediction


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        prediction, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        prediction = self.predict_labels_binary(distances)
        return prediction

