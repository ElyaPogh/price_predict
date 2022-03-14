from google.cloud import bigquery
import google.cloud.bigquery as bq

import pandas as pd
import numpy as np
import math

import helpers
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

from configparser import ConfigParser



class DBConnect():
    """DBConnect class aims to connect BigQuery database and table
     with provided credentials, it can load the table and save it as csv file,
      then read as a DataFrame
    """

    def __init__(self, credential_path, dataset_name, table_name):
        """

        Args:
            credential_path: credential_path the directory where credentials.json file is located
            dataset_name: dataset_name the name of the dataset
            table_name: table_name name of the table from the dataset
        """
        self.dataframe = pd.DataFrame()
        self.credential_path = credential_path
        self.dataset_name = dataset_name
        self.table_name = table_name
        self.path_to_save = None

    def save(self, path_to_save):
        """

        Args:
            path_to_save:  where the dataframe should be saved as csv


        """
        self.path_to_save = path_to_save
        client = bigquery.Client.from_service_account_json(self.credential_path)
        table = bigquery.TableReference.from_string(f'{self.dataset_name}.{self.table_name}')
        rows = client.list_rows(table)
        df = rows.to_dataframe()
        df.to_csv(self.path_to_save, index=False)

    def load_dataframe(self):
        """

        Returns: DataFrame for the saved table

        """
        if self.path_to_save:
            self.dataframe = pd.read_csv(self.path_to_save)
        return self.dataframe

    def get_dataframe(self):
        """

        Returns: Dataframe for the saved table

        """
        return self.dataframe


class Preprocess():
    """
    The preprocess class has purpose to do feature engineering, feature extraction, feature scaling,
    encode categorical variables, remove outliers if there exist ones and prepare data for input to neural network.
    """

    def __init__(self, dataframe):
        """

        Args:
            dataframe: The loaded dataframe from DBConnect class
        """
        self.dataframe = dataframe

    def feature_engineering(self):
        """
        The feature engineering method has been created after visualizing the data. Some features have been dropped
        because of not being correlating to the dependent variable(price). Address and Date columns have been changed
        and new features have been derived from them. Several methods have been implemented for categorical encoding,
        depending on the speciality of the feature.

        The Returns: preprocessed dataframe for neural network modelling

        """

        ord_enc = OrdinalEncoder()
        sd_scaler = StandardScaler()

        pp_dataframe = self.dataframe.copy()
        pp_dataframe['date'] = pd.to_datetime(pp_dataframe['date']).dt.date  # making changes in the date column to
        # take only months from it. As the time series plot showed, there is a pattern detected between prices and
        # months
        pp_dataframe['month'] = pd.DatetimeIndex(pp_dataframe['date']).month
        month_sin, month_cos = helpers.transformation(pp_dataframe['month'])  # the recommended encoding for cyclical
        # continous features
        pp_dataframe[[
            'month_sin',
            'month_cos']] = pd.Series([month_sin, month_cos])

        pp_dataframe[['type', 'tenure']] = pp_dataframe[['type', 'tenure']].astype('category')  # applying ordinal
        # encoding for type and tenure features
        pp_dataframe[['type_cat', 'tenure_cat']] = ord_enc.fit_transform(pp_dataframe[['type', 'tenure']])

        # extracting those string which include the mentioned keywords to specify the street
        pp_dataframe[['roads', '1']] = pp_dataframe['address'].str.lower().str.extract(
            r'\d{0,4}([A-z\s]+(street|avenue|road|place|square|lane|close|yard|market|walk|court))', expand=True)

        pp_dataframe = pp_dataframe.dropna()
        others = []
        for key, value in pp_dataframe[
            'roads'].value_counts().items():  # taking those roads which occur more than 50 times in the dataframe, and renaming to Other the rest
            if value < 50:
                others.append(key)
        pp_dataframe.loc[pp_dataframe['roads'].isin(others), 'roads_new'] = 'other'
        pp_dataframe.loc[~pp_dataframe['roads'].isin(others), 'roads_new'] = pp_dataframe['roads']

        dummies = pd.get_dummies(pp_dataframe['roads_new'])  # getting dummy variables for those streets
        pp_dataframe = pd.concat([pp_dataframe, dummies], axis=1)

        to_drop = ['date', 'roads', '1', 'area', 'month', 'roads_new', 'type',
                   'tenure', 'address']

        pp_dataframe = pp_dataframe.drop(to_drop, axis=1)  # dropping unnecessary features
        if 'price' not in pp_dataframe:
            pp_dataframe['price'] = None
        pp_dataframe[['longitude', 'latitude', 'price']] = sd_scaler.fit_transform(
            pp_dataframe[['longitude', 'latitude', 'price']].astype(np.float))
        pp_dataframe[['longitude', 'latitude', 'price']] = sd_scaler.fit_transform(
            pp_dataframe[['longitude', 'latitude', 'price']].astype(np.float))

        return pp_dataframe


class Modeling():
    """
    The Modeling class is created to split preprocess dataframe to train and test segments, to train ANN to solve the
    regression problem of predicting house prices with the selected features
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.path = None
        self.model_loaded = None

    def get_data_prepared(self, dataframe):
        """

        Args:
            dataframe: preprocessed dataframe

        Returns: split data, ready for training and testing

        """
        self.X = dataframe.loc[:, dataframe.columns != 'price']
        self.y = dataframe['price']
        X_arr = np.asarray(self.X).astype('float32')
        y_arr = np.asarray(self.y).astype('float32')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_arr, y_arr, test_size=0.25,
                                                                                random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_model(self):
        """

        For the model the hyperparameter tuning has been done with grid search for epochs and batch sizes. Also
        the model has been trained with sigmoid activation function,but with relu it demonstrated better results.

        """
        self.model = Sequential()
        self.model.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(320, activation='relu'))
        self.model.add(Dense(440, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='Adam', loss='mse')

    def train_model(self):
        """

        Returns: trained model and overall summary of the model

        """
        self.model.fit(x=self.X_train, y=self.y_train,
                       validation_data=(self.X_test, self.y_test),
                       batch_size=250,
                       epochs=45)
        print(f'Train loss is {self.model.evaluate(self.X_train, self.y_train)}',
              f'Test loss is {self.model.evaluate(self.X_test, self.y_test)}')

        return self.model, self.model.summary()

    def predict_model(self):
        """

        Returns: predicted values for X_train

        """
        predictions = self.model.predict(self.X_train)
        preds_list = [helpers.back_to_back(i, self.y) for i in predictions]
        return preds_list

    def save_config(self, path):
        """

        Args:
            path: Path to save the config

        Returns:

        """
        config = ConfigParser()
        config.read('config.ini')
        config.add_section('main')
        config.set('main', 'df_std', self.y.std())
        config.set('main', 'df_mean', self.y.mean())

        with open(path, 'w') as f:
            config.write(f)

    def save_model(self, path):
        """

        Args:
            path: directory where the model should be saved


        """

        self.model.save(path)

    def load_model(self, path):
        """

        Args:
            path: directory from where the model should be loaded

        """
        self.model_loaded = tf.keras.models.load_model(path)
        return self.model_loaded


