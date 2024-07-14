import time

import pandas as pd

from utils.band_filter_ml import get_spectra_filtered_list
from ML_algorithm.pre_process_pipeline import Pipeline
from ML_algorithm.Log_reg import logistic_regression_algorithm
from ML_algorithm.Naive_Bayes import naive_bayes_algorithm
from ML_algorithm.SVM import svm_algorithm
from ML_algorithm.KNN import KNN_algorithm
from config import UserInput


def main() -> None:
    start_time = time.perf_counter()
    input_df = pd.read_csv(UserInput.INPUT_PATH, sep=';')

    # Filtering
    spectral_ml_list = get_spectra_filtered_list(UserInput.SPECTRAL_DB_PATH, UserInput.BAND_DISTANCE_CHECK, 
                                                 [UserInput.ANALYSIS_COMPOUND_NAME, input_df], cpu_cores=UserInput.CPU_CORES)
    
    # Machine Learning
    normalized_input_dataframe, normalized_database_dataframe = Pipeline(spectral_ml_list, UserInput.ANALYSIS_COMPOUND_NAME).get_split_dataframes()

    KNN_algorithm((normalized_input_dataframe, normalized_database_dataframe), verbose=True)
    #svm_algorithm((normalized_input_dataframe, normalized_database_dataframe), verbose=True)
    logistic_regression_algorithm((normalized_input_dataframe, normalized_database_dataframe), verbose=True)
    naive_bayes_algorithm((normalized_input_dataframe, normalized_database_dataframe), verbose=True)

    print(f'Complete execution time: {time.perf_counter() - start_time} seconds.')

if __name__ == "__main__":
    main()