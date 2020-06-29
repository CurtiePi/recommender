import re
import os
from requests import get
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from contextlib import closing
import pprint
import pandas as pd
import numpy as np

import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#%matplotlib inline

from oauth2client import file, client, tools
from googleapiclient.discovery import build
from googleapiclient import discovery
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import httplib2


class Recommender():

    def __init__(self):
        pass

    def process_k_nearest(self, dataframe):
        data_cols = ['STR', 'DEX', 'CON', 'INT', 'WIS', 'CHA']
        X = dataframe[data_cols].values

        y = dataframe['target'].values

        # Training data split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
        #print(f"X training shape: {X_train.shape} - y training shape: {y_train.shape}")
        #print(f"X testing shape: {X_test.shape} - y testing shape: {y_test.shape}")

        # Use k nearest neighbor
        Ks = 11
        mean_acc = np.zeros((Ks - 1))
        std_acc = np.zeros((Ks - 1))
        k_neigh_list = []
        for k in range(1, Ks):
            neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)

            k_neigh_list.append(neigh)
            # Predict based on k
            y_hat = neigh.predict(X_test)
            mean_acc[k - 1], std_acc[k - 1] = evaluate_accuracy(k, neigh, X_train, y_train, y_test, y_hat)

        print(f"The best accuracy is with {mean_acc.max()} and k equal to {mean_acc.argmax() + 1}")
        return k_neigh_list[mean_acc.argmax()]


    def process_new_data(self, input_data, k_neigh, map):
        data_cols = ['STR', 'DEX', 'CON', 'INT', 'WIS', 'CHA']
        D = input_data[data_cols].values
        Z = D.reshape(1, -1)
        prediction = k_neigh.predict(Z)
        print(f"With these attribute scores we recommend the class: {map[prediction[0]]}")


    def evaluate_accuracy(self, k, neighbors, train_X, train_y, test_y, y_hat):
        #print(f"Accuracy with {k} nearest neighbors:\n")
        #print(f"Train set Accuracy: {metrics.accuracy_score(train_y, neighbors.predict(train_X))}")
        mean_score = metrics.accuracy_score(test_y, y_hat)
        standard_deviation = np.std(y_hat == test_y)/np.sqrt(y_hat.shape[0])
        #print(f"Mean Accuracy: {mean_score}")
        #print(f"Standard Deviation Accuracy: {standard_deviation}\n\n")

        return mean_score, standard_deviation




def load_file_data(input_data):
    df = pd.read_csv("character_data.csv")
    w_df = df.copy(True)
    surplus_cols = ['STR_BONUS', 'DEX_BONUS', 'CON_BONUS', 'INT_BONUS', 'WIS_BONUS', 'CHA_BONUS', 'SUB_RACE', 'CLASS_TWO']
    w_df = w_df.drop(surplus_cols, axis=1)
    w_df['CLASS_ONE'] = w_df['CLASS_ONE'].astype('category')
    w_df['RACE'] = w_df['RACE'].astype('category')
    w_df['target'] = w_df['CLASS_ONE'].cat.codes
    map = dict( enumerate(w_df['CLASS_ONE'].cat.categories))

    w_df = w_df.drop(['RACE', 'CLASS_ONE'], axis=1)

    normalize_cols = ['STR', 'DEX', 'CON', 'INT', 'WIS', 'CHA']
    
    new_data_dict = {normalize_cols[n]: input_data[n] for n in range(0, len(input_data))}
    new_df = normalize(w_df.drop('target', axis=1).append(new_data_dict, ignore_index=True).tail(), normalize_cols)
    new_df = new_df.iloc[-1]

    norm_df = normalize(w_df, normalize_cols)

    return norm_df, new_df, map



def normalize(df, n_cols):
    for col in n_cols:
        max_val = df[col].max()
        min_val = df[col].min()
        df[col] = (df[col] - min_val)/(max_val - min_val)

    return df


def main(new_data):
   #print("Getting the data from the spreadsheet")
   #data = get_googlesheets_data()
   #print("Putting the data into a panda dataframe")
   #dataframe = create_dataframe(data)
   df, new_df, map = load_file_data(new_data)
   best_k = process_data(df)
   process_new_data(new_df, best_k, map)


if __name__ == '__main__':
    import sys
    data = [int(x) for x in sys.argv[1:]]
    #print(data)
    main(data)
