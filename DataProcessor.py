import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
import numpy as np
import pandas as pd


class DatasetProcessor:

    def __init__(self, pop, fitness):

        self.pl = None
        self.pop = pd.DataFrame(pop)
        self.fitness = pd.DataFrame(fitness)
        self.augmented_XTrain, self.augmented_XTest, self.augmented_YTrain, self.augmented_YTest, self.X_train, self.X_test, self.y_train, self.y_test = self.dataAugmentation()
        # self.augmented_XTrain, self.augmented_XTest, self.augmented_YTrain, self.augmented_YTest, self.X_train,
        # self.X_test, self.y_train, self.y_test = self.dataAugmentation(X_train, X_test, y_train, y_test)

    def dataNormalization(self, df):

        normalizedData = (df - df.min()) / (df.max() - df.min())
        return normalizedData

    def split_data(self, x, y):
        return train_test_split(x, y, test_size=0.20, random_state=42)

    @staticmethod
    def Decision_Variable_Augmentation(X_train, X_test):
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        num_rows = X_train.shape[0] * X_test.shape[0]
        num_cols = X_train.shape[1] + X_test.shape[1]

        new_dec_var = [[0] * num_cols for _ in range(num_rows)]
        idx = 0
        for i in range(len(X_test)):
            for j in range(len(X_train)):
                for k in range(num_cols):
                    if k < len(X_train.columns):
                        new_dec_var[idx][k] = X_test.iat[i, k]
                    else:
                        new_dec_var[idx][k] = X_train.iat[j, k - len(X_train.columns)]
                idx += 1
        return pd.DataFrame(new_dec_var, columns=["X" + str(i) for i in range(1, num_cols + 1)])

    def Objective_Augmentation(self, y_train, y_test):
        num_rows = y_train.shape[0] * y_test.shape[0]

        new_obj = [[0] for _ in range(num_rows)]

        idx = 0
        for i in range(len(y_test)):
            for j in range(len(y_train)):
                new_obj[idx] = y_test.iat[i, 0] - y_train.iat[j, 0]
                idx += 1

        return pd.DataFrame(new_obj, columns=['F1'])

    def dataAugmentation(self):

        #Loading Excel File and creating dataframe
        #df = pd.read_excel(self.data_dir, sheet_name=self.sheet_names[self.problem_index])

        #counting Decision variable columns
        #blank_col_index = df.columns[df.isnull().all()].tolist()
        #if blank_col_index:
        ##else:
        # blank_col_index = len(df.columns) - 1

        #X_columns = df.columns[:blank_col_index]
        #f_columns = df.columns[blank_col_index + 1:blank_col_index + 2]

        #splitting data into Decision variable and Objective Function Values
        #features = df[X_columns]
        #obj = df[f_columns]

        #Data Normalization
        dec_val = self.pop
        obj = self.dataNormalization(self.fitness)

        #Splitting Data into Train and Test
        X_train, X_test, y_train, y_test = self.split_data(dec_val, obj)

        temp = [[0] for _ in range(len(y_test))]
        for i in range(len(y_test)):
            temp[i] = y_test.iat[i, 0]
        y_test = pd.DataFrame(temp)

        temp = [[0] * X_test.shape[1] for _ in range(len(X_test))]
        for i in range(len(X_test)):
            for j in range(X_test.shape[1]):
                temp[i][j] = X_test.iat[i, j]
        X_test = pd.DataFrame(temp)

        #Data Augmentation
        augmented_XTrain = self.__class__.Decision_Variable_Augmentation(X_train, X_train)
        augmented_XTest = self.__class__.Decision_Variable_Augmentation(X_train, X_test)
        augmented_YTrain = self.Objective_Augmentation(y_train, y_train)
        augmented_YTest = self.Objective_Augmentation(y_train, y_test)

        return augmented_XTrain, augmented_XTest, augmented_YTrain, augmented_YTest, X_train, X_test, y_train, y_test

    def cls(self, y_train, y_test):
        num_rows = y_train.shape[0] * y_test.shape[0]
        cls_obj = [[0] for _ in range(num_rows)]
        idx = 0
        for i in range(len(y_test)):
            for j in range(len(y_train)):
                if y_test.iat[i, 0] - y_train.iat[j, 0] == 0:
                    cls_obj[idx] = 0
                elif y_test.iat[i, 0] - y_train.iat[j, 0] > 0:
                    cls_obj[idx] = 1
                else:
                    cls_obj[idx] = 2
                idx += 1
        return pd.DataFrame(cls_obj)

    @staticmethod
    def forClassification(y_train, predicted_YTest):
        col = 3
        row = y_train.shape[0]
        pl = [[0] * col for _ in range(row)]
        idx = 0
        for j in range(len(y_train)):
            for k in range(col):
                if k == 0:
                    pl[idx][k] = j + 1
                elif k == 1:
                    pl[idx][k] = y_train.iat[j, 0]
                else:
                    pl[idx][k] = predicted_YTest[idx]
            idx += 1
        pl_df = pd.DataFrame(pl, columns=['samples', 'values', 'class'])

        # Get the indices where predicted_ytest is 1
        indices_1 = pl_df[pl_df['class'] == 1].index

        # Find the rows where predicted_YTest is 2

        indices_2 = pl_df[pl_df['class'] == 2].index

        # Find the max value of y_train where predicted_YTest is 1
        if not indices_1.empty:
            max_value = pl_df.loc[indices_1, 'values'].max()
        else:
            max_value = None  # No 1s found in predicted_YTest

        # Find the min value of y_train where predicted_YTest is 2
        if not indices_2.empty:
            min_value = pl_df.loc[indices_2, 'values'].min()
        else:
            min_value = None  # No 2s found in predicted_YTest

        # Calculate the mean based on available values
        if max_value is not None and min_value is not None:
            mean = ((max_value) + (min_value)) / 2
        elif max_value is not None:
            mean = max_value  # If all values in predicted_YTest are 1
        elif min_value is not None:
            mean = min_value  # If all values in predicted_YTest are 2
        else:
            mean = 0  # If no valid values are found

        return mean
