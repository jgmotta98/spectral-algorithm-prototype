from multiprocessing import cpu_count
import time

import pandas as pd

from utils.band_filter import get_spectra_filtered_list
from standard_algorithm.similarity_algorithm import local_algorithm, KNN_algorithm, svg_algorithm, naive_bayes_algorithm, logistic_regression_algorithm
from utils.generate_report import create_graph


SPECTRAL_DB_PATH = '.\\files\\database\\spectral_database.db'
BAND_DISTANCE_CHECK = 25 # Entre 10 e 40 -> Com o ideal sendo 25.
CPU_CORES = cpu_count()
OUTPUT_PDF = '.\\files\\reports\\report_example_Bitertanol.pdf'
DF_PATH = '.\\files\\csv_tests\\Bitertanol.csv'
ANALYSIS_COMPOUND = 'Bitertanol_teste'
PICKLE = '.\\files\\database\\teste.pickle'


def get_input_df(df_path: str):
    df = pd.read_csv(df_path, sep=';')
    return df


def main() -> None:
    start_time = time.perf_counter()
    input_df = get_input_df(DF_PATH)
    spectral_filtered_list, spectral_input_list, spectral_list = get_spectra_filtered_list(SPECTRAL_DB_PATH, BAND_DISTANCE_CHECK, [ANALYSIS_COMPOUND, input_df], cpu_cores=CPU_CORES)
    #svg_algorithm(pd.read_pickle(PICKLE), ANALYSIS_COMPOUND)
    #KNN_algorithm(pd.read_pickle(PICKLE), ANALYSIS_COMPOUND)
    #logistic_regression_algorithm(pd.read_pickle(PICKLE), ANALYSIS_COMPOUND)
    #naive_bayes_algorithm(pd.read_pickle(PICKLE), ANALYSIS_COMPOUND)
    components_data_filter, input_list_dict, components_result = local_algorithm(spectral_filtered_list, spectral_input_list, ANALYSIS_COMPOUND)
    create_graph(components_data_filter, input_list_dict, spectral_list, input_df, components_result, ANALYSIS_COMPOUND, OUTPUT_PDF, BAND_DISTANCE_CHECK)
    print(f'Complete execution time: {time.perf_counter() - start_time} seconds.')

if __name__ == "__main__":
    main()