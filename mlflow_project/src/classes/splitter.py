from classes.intermediate_step import IntermediateStep
from sklearn.model_selection import train_test_split

import pandas as pd
import os

class Splitter(IntermediateStep):
    def __init__(
            self, name, data_path, date_cols, target_variable, destination_directory, dates_data_path, column_to_split_by, test_size, random_state
    ):

        super().__init__(name, data_path, date_cols, target_variable, destination_directory)
        self.dates_data_path = dates_data_path
        self.column_to_split_by = column_to_split_by
        self.dates_data = None
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def set_train_test(self, X, y , test_size, random_state):
       
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        print(f"""Test and train attributes defined {test_size}.
        Test size: {len(self.X_test)}
        Train size: {len(self.X_train)}""")


    def save_data(self, destination_directory):
            os.makedirs(destination_directory, exist_ok=True)
            self.X_train.to_csv(destination_directory + '/X_train.csv')
            self.X_test.to_csv(destination_directory + '/X_test.csv')
            self.y_train.to_csv(destination_directory + '/y_train.csv')
            self.y_test.to_csv(destination_directory + '/y_test.csv')
            print(f"Data saved to {destination_directory}")


    def execute(self):
        print(f"-------------- Executing {self.name} --------------")
        self.load_data(self.data_path, self.date_cols, index_col=0)
        self.load_dates_data(self.dates_data_path, self.date_cols, index_col=0)
        self.set_train_test(self.data.loc[:, self.data.columns != self.target_variable]
                            , self.data[self.target_variable]
                            , self.test_size
                            , self.random_state)
        #self.save_data(self.destination_directory)
        print(f"--------------- {self.name} finished ---------------")

    # These methods below are specific for this use case
    def load_dates_data(self, data_path, date_cols, index_col=None):
        self.dates_data = pd.read_csv(data_path, parse_dates=date_cols, index_col=index_col
                                      )
        print(f"Dates data loaded from {data_path}")


    def _add_date_column_to_data(self, data, dates_data_path, column_to_split_by):
        """This function adds a column with the date to the data
        """
        # Load the dates data
        dates_data = pd.read_csv(dates_data_path, index_col=0)
        dates_data = dates_data[[column_to_split_by]]
        data = data.merge(
            dates_data, how='left', left_index=True, right_index=True)
        data[self.column_to_split_by] = data[self.column_to_split_by].astype(
            'datetime64[ns]')
        
        print(f"Date column {column_to_split_by} added to the data")
        return data

    def _filter_by_month(self, data, from_month, to_month, column_to_split_by):
        """This function filters the data from month to month since the first date
        """
        cutoff_lower = data[column_to_split_by].min(
        ) + pd.DateOffset(months=from_month)
        cutoff_upper = data[column_to_split_by].min(
        ) + pd.DateOffset(months=to_month)

        return data[(data[column_to_split_by] >= cutoff_lower) & (data[column_to_split_by] < cutoff_upper)]

    def x_y_filter_by_month(self, from_month, to_month):
        """This method returns the X and y data for the next period
        """
        filtered_data = self._add_date_column_to_data(self.data
                                      , self.dates_data_path
                                      , self.column_to_split_by)
        
        filtered_data = self._filter_by_month(filtered_data
                                              , from_month
                                              , to_month
                                              , self.column_to_split_by)
        filtered_data = filtered_data.drop(self.column_to_split_by, axis=1)
        print (f"Data filtered by {from_month} and {to_month} months")
        return filtered_data.loc[:, filtered_data.columns != self.target_variable], filtered_data[self.target_variable]


    def set_train_test_filtered(self, number_of_months):
        X, y=  self.x_y_filter_by_month(0, number_of_months)

        self.set_train_test(X
                            , y
                            , self.test_size
                            , self.random_state
                            )
    