import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from DataProcessor import DatasetProcessor
import matplotlib.pyplot as plt


class MLPClassifierModel:
    error_list = []  # To store the error for each training session

    def __init__(self, dataset_processor: DatasetProcessor):
        """
        Initialize MLPClassifierModel with a DatasetProcessor instance.
        """
        # Store the dataset_processor instance to access its methods and data
        self.dp = dataset_processor  # DatasetProcessor instance
        self.predicted_YTest = None

        # Initialize the MLPClassifier model
        self.model = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=100, alpha=0.0001,
                                   solver='sgd', verbose=10, random_state=21, tol=1e-9)

        # Access the required data via the DatasetProcessor instance
        self.YTrain_class = self.dp.cls(self.dp.y_train, self.dp.y_train)
        self.YTest_class = self.dp.cls(self.dp.y_train, self.dp.y_test)

    def train(self):
        """Train the model using data from the DatasetProcessor and save the trained model."""
        # Use augmented_XTrain from DatasetProcessor
        self.model.fit(self.dp.augmented_XTrain, self.YTrain_class)
        # Save the model after training
        joblib.dump(self.model, 'trained_mlp_classifier.pkl')
        print("Model trained and saved as 'trained_mlp_classifier.pkl'")
        self.predict_with_saved_model()

    def predict_with_saved_model(self, Test=None):
        """Load the saved model and predict values for the test set."""
        # Load the saved model from disk
        self.model = joblib.load('trained_mlp_classifier.pkl')
        y_predict = []
        # Use Test from DatasetProcessor if not provided
        if Test is not None:
            # Predict using the loaded model
            self.predicted_YTest = self.model.predict(Test)
            return self.predicted_YTest
        else:
            Test = self.dp.augmented_XTest
            # Predict using the loaded model
            self.predicted_YTest = self.model.predict(Test)
            # Use the saved model to predict
            for i in range(0, len(self.predicted_YTest), len(self.dp.y_train)):
                # Slice y_train and predicted_YTest for this batch
                predicted_YTest_batch = self.predicted_YTest[i:i + len(self.dp.y_train)].reshape(-1, 1)

                # Call forClassification on the current batch
                mean = self.dp.forClassification(self.dp.y_train, predicted_YTest_batch)

                # Append the result to the means list
                y_predict.append(mean)

            # Calculate Mean Squared Error
        mse = mean_squared_error(self.dp.y_test, y_predict)
        self.error_list.append(mse)  # Store the error in thike error_list

    def retrain_model(self):
        """Retrain the model with new data and save the updated model."""
        # Load the previously trained model
        self.model = joblib.load('trained_mlp_classifier.pkl')
        print("Model loaded for retraining from 'trained_mlp_classifier.pkl'")
        # Retrain the model with new training data
        self.model.fit(self.dp.augmented_XTrain, self.YTrain_class)
        print("Model retrained with new data")

        # Save the updated model
        joblib.dump(self.model, 'trained_mlp_classifier_retrained.pkl')
        print("Retrained model saved as 'trained_mlp_classifier_retrained.pkl'")

        # Predict again with the retrained model
        self.predict_with_saved_model()

    # Assuming self.error_list contains the MSE values for each generation

    @staticmethod
    def plot_error_vs_generation():
        return MLPClassifierModel.error_list

    # Call this method in your class after your training completes
