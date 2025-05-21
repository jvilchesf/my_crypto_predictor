import pandas as pd
import mlflow
import great_expectations as gex
from loguru import logger


class SplitData:
    def __init__(self, ts_data: pd.DataFrame):
        self.ts_data = ts_data

    def split_train_test_datasets(self, train_test_split_ratio) -> pd.DataFrame:
        train_size = int(len(self.ts_data) * train_test_split_ratio)

        train_data = self.ts_data.iloc[:train_size]
        test_data = self.ts_data.iloc[train_size:]
        # Log parameters into mlflow
        mlflow.log_param("train_size", train_data.shape)
        mlflow.log_param("test_size", test_data.shape)

        # Split data into features and target
        X_train = train_data.drop(columns=["target"])
        y_train = train_data["target"]
        X_test = test_data.drop(columns=["target"])
        y_test = test_data["target"]

        # log parameters into mlflow
        mlflow.log_param("X_train_shape", X_train.shape)
        mlflow.log_param("Y_train_shape", y_train.shape)

        return X_train, y_train, X_test, y_test

    def split_test_compare_datasets(
        self, X_test: pd.DataFrame, y_test: pd.DataFrame
    ) -> pd.DataFrame:
        """ "
        This function split takes 50% of rows from the test dataframe to create a compare dataframe
        This compare dataframe is for compare top n models returned by lazy predictor
        It will return a new test dataframe with the last 50% of the rows that it originally had

        Args:
            X_test: Dataframe with test row data
            y_test: Dataframe with test target values

        Return:
            X_test_compare: New Dataframe with 50% of original test dataframe, it'll be use to compare different models
            y_test_compare: New Dataframe with 50% of original target test dataframe, it'll be use to compare different models
            X_test: Dataframe with test data to evaluate the best model selected
            y_test: Dataframe with target data to evaluate the best model selected
        """

        # Split test dataset into 50% 50% for comparing models purposes
        X_test_50_percent_rows = int(len(X_test) / 2)
        X_test_compare = X_test[:X_test_50_percent_rows]
        y_test_compare = y_test[:X_test_50_percent_rows]
        X_test = X_test[X_test_50_percent_rows:]
        y_test = y_test[X_test_50_percent_rows:]

        # Log new dataframes in mlflow
        mlflow.log_param("X_test_shape", X_test.shape)
        mlflow.log_param("Y_test_shape", y_test.shape)
        mlflow.log_param("X_test_compare", X_test_compare.shape)
        mlflow.log_param("y_test_compare", y_test_compare.shape)

        return X_test_compare, y_test_compare, X_test, y_test


class TransformCleanData:
    def __init__(self, ts_data: pd.DataFrame):
        self.ts_data = ts_data

    def validate_data(self, treshold_null_values: float):
        """
        Runs a bunch of validations, if any of them fail, the function will raise an exception
        """

        # Check if the numeric columns are positive
        ge = gex.from_pandas(self.ts_data)

        # validation_results = ge.expect_column_values_to_be_between(column='open', min_value=0, max_value=float('inf'))
        # validation_results = ge.expect_column_values_to_be_between(column='high', min_value=0, max_value=float('inf'))
        # validation_results = ge.expect_column_values_to_be_between(column='low', min_value=0, max_value=float('inf'))
        validation_results = ge.expect_column_values_to_be_between(
            column="close", min_value=0, max_value=float("inf")
        )
        # validation_results = ge.expect_column_values_to_be_between(column='volume', min_value=0, max_value=float('inf'))

        # - Check for datetime corrected format
        # - Check for duplicate rows
        # - Check data is sorted by window_start_ms

        if not validation_results.success:
            raise Exception("Data validation failed")

        num_rows_df_before_clean = len(self.ts_data)
        # Validate null values
        rows_null_values = sum(self.ts_data.isnull().sum())
        prcentage_dataset_null_values = rows_null_values / num_rows_df_before_clean
        if prcentage_dataset_null_values > treshold_null_values:
            raise Exception(
                f"Dataset has too many null values: {prcentage_dataset_null_values:.2%} exceeds threshold of {treshold_null_values:.2%}"
            )
        else:
            self.ts_data = self.ts_data.dropna()
            logger.info(
                f"Dropped {num_rows_df_before_clean - len(self.ts_data)} rows with null values"
            )

        return self.ts_data
