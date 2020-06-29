import re
import pandas as pd
import numpy as np


class DataFrameBucket():

    """

        Formats data into a dataframe for use by model

        file_name (str): Name of file containing data
    """

    def __init__(self, file_name):
        self._dataframe = pd.read_csv(file_name)
        self._map = {}


    def set_required_columns(self, required_cols):
        """
            
            Maniputlate dataframe to only have required columns

            required_cols (list): names of columns required for dataframe
        """

        colunms = self._dataframe.columns
        drop_cols = [col_name for col_name in columns if col_name not in required_cols]
        self._dataframe = self._dataframe.drop(drop_cols, axis=1)


    def convert_category_columns(self, category_cols, map_cols):
        """
            
            Convert dataframe columns to type category

            category_cols (list): names of columns that should be type category
 
            map_cols(list): names of columns that need to map back to original value
        """

        convert_dict = {col_name: 'category for col_name in category_cols}
        self._dataframe = self._dataframe.astype(convert_dict)

        for col_name in map_cols:
            self._map[col_name] = dict(enumerate(self._dataframe[col_name].cat.categories))


    def set_tartget_column(self, target_col):
        """

            Set the target column for the dataframe

            target_col (str): name of the column to set as the target column
        """

        self._dataframe['target'] = self._dataframe[target_col].cat.codes



    def get_normalized_dataframe(self, normalize_cols):
        """

            Normalizes the dataframe

            normalize_cols (list): name of the columns on which to normalize
        """

        return self.normalize(self._dataframe, normalize_cols)
    

    def format_input_data(self, input_data, data_cols):
        """

            Incorporate new data into a normalized dataframe format 

            input_data (list): list of data that serves as input data

            data_cols (list): name of the columns that serve as input
        """

        new_data_dict = {data_cols[n]: input_data[n] for n in range(0, len(input_data))}
        new_df = self._dataframe.drop('target', axis=1).append(new_data_dict, ignore_index=True).tail()
        input_dataframe = self.normalize(new_df, data_cols)
        input_dataframe = input_dataframe.iloc[-1]

        return input_data_dataframe


    def drop_columns(self, cols_to_drop):
        """
            
            Maniputlate dataframe to drop specific columns

            cols_to_drop (list): names of columns to be dropped from dataframe
        """
        self._dataframe = self._dataframe.drop([cols_to_drop], axis=1)


    def map_category_cols(self, col_name):
        """

            Create a map of original values to category values

            col_name (str): Name of the column to be mapped
        """
        self._map = dict(enumerate(self._dataframe[col_name].cat.categories))


    def normalize(self, df, n_cols):
        """

            Create a map of original values to category values

            df (Dataframe): dataframe object

            n_cols (list): Names of the column to be normalized
        """
        for col in n_cols:
            max_val = df[col].max()
            min_val = df[col].min()
            df[col] = (df[col] - min_val)/(max_val - min_val)

        return df
