from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class Pipeline:
    def __init__(self, df: pd.DataFrame, input_comp_name: str) -> None:
        self.df = df
        self.input_comp_name = input_comp_name

        self.min_max_scaler()

    def min_max_scaler(self):
        scaler = MinMaxScaler()
        for col in self.df.columns:
            if col != 'Name':
                scaled_column = scaler.fit_transform(self.df[[col]])
                self.df[col] = scaled_column.flatten()
            else:
                self.df[col] = self.df[col]

    def get_split_dataframes(self):
        test_df = self.df[self.df['Name'] == self.input_comp_name]
        test_df.reset_index(inplace=True, drop=True)

        train_df = self.df[self.df['Name'] != self.input_comp_name]
        train_df.reset_index(inplace=True, drop=True)

        return test_df, train_df