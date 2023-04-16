from classes.step import Step
from typing import Union
import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

class Cleaner(Step):
    def __init__(
            self
            , name
            , raw_data_path
            , columns_to_keep
            , destination_directory
            , dict_columns_dtypes
            , null_threshold
            , max_corr
            ):
        super().__init__(name)
        self.raw_data_path = raw_data_path
        self.columns_to_keep = columns_to_keep
        self.destination_directory = destination_directory
        self.dict_columns_dtypes = dict_columns_dtypes
        self.null_threshold = null_threshold
        self.max_corr = max_corr
        self.data = None

    def load_data(self, raw_data_path):
        df = pd.read_csv(raw_data_path)
        print(f"Data loaded from {raw_data_path}")
        self.data = df
    
    def _keep_columns(self, columns_to_keep):
        self.data = self.data[columns_to_keep]
        print(f"Columns kept {columns_to_keep}")

    def check_data(self):
        """
        Check the data types and percentage of null values in a Pandas DataFrame.

        Args:
        ----
            df : pd.DataFrame
                Input DataFrame to check

        Returns:
        -------
            pd.DataFrame
                DataFrame containing data types and null percentages for each column.
        """
        dtypes = self.data.dtypes
        null_percentages = self.data.isnull().mean() * 100
        check_df = pd.DataFrame(
            {'Data Types': dtypes, 'Null Percentages': null_percentages})
        return check_df .join(self.data.head(3).T)

    def _convert_column_to_datatype(self, column: str, dtype: Union[type, str]):
        """
        Converts a Pandas DataFrame column to a given data type.

        Args:
        ----
            df : pd.DataFrame
                Input DataFrame to modify
            column : str
                Column name to convert to the given data type
            dtype : type or str
                Data type to convert the column to, either as a Python type or a string
                (e.g., 'float', 'int', 'datetime64', etc.)

        Returns:
        -------
            pd.DataFrame
                Modified DataFrame with the specified column converted to the given data type.
        """
        self.data[column] = self.data[column].astype(dtype)
        print(f"Column {column} converted to {dtype}")
    
    def _convert_columns_to_datatype(self, dict_columns_dtypes: dict):
        """
        Converts a list of Pandas DataFrame columns to a given data type.

        Args:
        ----
            

        Returns:
        -------
            pd.DataFrame
                Modified DataFrame with the specified columns converted to the given data type.
        """

        for column, dtype in dict_columns_dtypes.items():
            self._convert_column_to_datatype(column, dtype)
       
    def _drop_columns_nulls(self, null_threshold: float):
        """
        Drops any column in a Pandas DataFrame with a percentage of nulls higher than a given threshold.

        Args:
        ----
            df : pd.DataFrame
                Input DataFrame to check
            threshold : float
                Threshold percentage of null values to drop columns. Must be a float between 0 and 100.

        Returns:
        -------
            pd.DataFrame
                DataFrame with dropped columns.
        """
        if not 0 <=  null_threshold <= 100:
            raise ValueError("Threshold must be a float between 0 and 100.")

        null_percentages = self.data.isnull().mean() * 100
        columns_to_keep = null_percentages[null_percentages <=  null_threshold].index.tolist(
        )
        self.data = self.data[columns_to_keep]
        print(f"Columns with null percentages higher than {null_threshold} dropped.")
    
    def _drop_high_correlation_vars(self, max_corr):
        corr_matrix = self.data.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(
            upper[column] > max_corr)]
        self.data = self.data.drop(to_drop, axis=1)
        print(f"Columns with correlation higher than {max_corr} dropped.")

    def corr_heatmap(self, title=None, figsize=(10, 10), annot=True, cmap='coolwarm', linewidth=.5, fontsize=7):

        plt.figure(figsize=figsize)
        df_corr = self.data.corr()
        df_corr = df_corr.round(2)
        mask = np.triu(np.ones_like(df_corr))
        sns.heatmap(df_corr, annot=annot, cmap=cmap, linewidth=linewidth,
                    mask=mask, annot_kws={"fontsize": fontsize})
        plt.title(title)
        plt.show()

    def save_data(self, destination_directory):
        os.makedirs(destination_directory, exist_ok=True)
        self.data.to_csv(destination_directory + '/clean_data.csv', index=False)
        print(f"Data saved to {destination_directory}")
    
    def execute(self):
        self.load_data(self.raw_data_path)
        self._keep_columns(self.columns_to_keep)
        self._convert_columns_to_datatype(self.dict_columns_dtypes)
        self._drop_columns_nulls(self.null_threshold)
        self._drop_high_correlation_vars(self.max_corr)
        self.save_data(self.destination_directory)