from multiprocessing import cpu_count
import time

import pandas as pd

from utils.band_filter import get_spectra_filtered_list
from standard_algorithm.similarity_algorithm import local_algorithm
from ML_algorithm.pre_process_pipeline import Pipeline
from ML_algorithm.Log_reg import logistic_regression_algorithm
from ML_algorithm.Naive_Bayes import naive_bayes_algorithm
from ML_algorithm.SVG import svg_algorithm
from ML_algorithm.KNN import KNN_algorithm
from utils.generate_report import create_graph


SPECTRAL_DB_PATH = '.\\files\\database\\spectral_database.db'
BAND_DISTANCE_CHECK = 25 # Entre 10 e 40 -> Com o ideal sendo 25.
CPU_CORES = cpu_count()
OUTPUT_PDF = '.\\files\\reports\\report_example_Endosulfan-sulfate.pdf'
DF_PATH = '.\\files\\csv_tests\\Endosulfan-sulfate.csv'
ANALYSIS_COMPOUND = 'Endosulfan-sulfate_teste'


def main() -> None:
    start_time = time.perf_counter()
    input_df = pd.read_csv(DF_PATH, sep=';')

    # Filtering
    spectral_filtered_list, spectral_input_list, spectral_list, spectral_ml_list = get_spectra_filtered_list(SPECTRAL_DB_PATH, BAND_DISTANCE_CHECK, 
                                                                                           [ANALYSIS_COMPOUND, input_df], cpu_cores=CPU_CORES)
    
    # Machine Learning
    normalized_input_dataframe, normalized_database_dataframe = Pipeline(spectral_ml_list, ANALYSIS_COMPOUND).get_split_dataframes()
    KNN_algorithm((normalized_input_dataframe, normalized_database_dataframe), verbose=True)
    svg_algorithm((normalized_input_dataframe, normalized_database_dataframe), verbose=True)
    logistic_regression_algorithm((normalized_input_dataframe, normalized_database_dataframe), verbose=True)
    naive_bayes_algorithm((normalized_input_dataframe, normalized_database_dataframe), verbose=True)

    # Authoral Algorithm
    components_data_filter, input_list_dict, components_result = local_algorithm(spectral_filtered_list, spectral_input_list, ANALYSIS_COMPOUND)

    # Report Generation
    create_graph(components_data_filter, input_list_dict, spectral_list, input_df, components_result, ANALYSIS_COMPOUND, OUTPUT_PDF, BAND_DISTANCE_CHECK)
    print(f'Complete execution time: {time.perf_counter() - start_time} seconds.')


if __name__ == "__main__":
    main()