from multiprocessing import cpu_count
import sqlite3
import time
from collections import OrderedDict
import multiprocessing as mp

import pandas as pd

from utils.band_filter import get_spectra_filtered_list
from utils.input_manipulation import input_baseline_correction
from utils.generate_report import create_graph
from standard_algorithm.similarity_algorithm import local_algorithm
from config import UserInput


def get_unique_spectral_names(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT DISTINCT SpectraNames.name 
        FROM SpectraNames
        INNER JOIN SpectraData ON SpectraNames.id = SpectraData.name_id
    ''')

    names = [row[0] for row in cursor.fetchall()]
    conn.close()
    return names


def fetch_spectral_data(db_path: str, name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT SpectraNames.name, SpectraData.wavelength, SpectraData.intensity 
        FROM SpectraNames 
        INNER JOIN SpectraData ON SpectraNames.id = SpectraData.name_id
        WHERE SpectraNames.name = ?
    ''', (name,))

    data = cursor.fetchall()
    conn.close()

    df = pd.DataFrame(data, columns=['name', 'x', 'y'])
    return df


def order_result_dict(result_list: list[dict[str, float]]) -> dict[str, float]:
    unified_dict: dict[str, float] = {}

    for d in result_list:
        unified_dict.update(d)

    final_result_dict = OrderedDict(sorted(unified_dict.items(), key=lambda x: x[1], reverse=True))
    final_result_dict = dict(list(final_result_dict.items())[:5])
    return final_result_dict


def get_filtered_data(spectral_list: list[pd.DataFrame], band_distance_check: int, 
                      spectral_filtered_input: tuple[pd.DataFrame, pd.DataFrame], get_input: bool = False) -> list[dict[str, float]]| list[dict[str, pd.DataFrame]]:
        return [local_algorithm(get_spectra_filtered_list(spectra, band_distance_check), 
                                spectral_filtered_input, 
                                get_dataframe=True, 
                                get_input=get_input) 
                                for spectra in spectral_list]


def convert_to_dict(data_list: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    result_dict: dict[str, pd.DataFrame]  = {}
    for data in data_list:
        result_dict.update(data)
    return result_dict


def paralelization_process(args) -> dict[str, float] | dict[str, pd.DataFrame]:
    name, spectral_filtered_input, spectral_db_path, band_distance_check = args
    # Fetch spectral data
    spectral_data = fetch_spectral_data(spectral_db_path, name)

    # Filtering
    spectral_filtered_database = get_spectra_filtered_list(spectral_data, band_distance_check)

    # Authoral Algorithm
    result_dict = local_algorithm(spectral_filtered_database, spectral_filtered_input, get_dataframe=False)

    return result_dict


def main() -> None:
    start_time = time.perf_counter()
    input_df = pd.read_csv(UserInput.INPUT_PATH, sep=';')
    unique_names = get_unique_spectral_names(UserInput.SPECTRAL_DB_PATH)

    spectral_input = input_baseline_correction([UserInput.ANALYSIS_COMPOUND_NAME, input_df])
    spectral_filtered_input = get_spectra_filtered_list(spectral_input, UserInput.BAND_DISTANCE_CHECK)

    if UserInput.USE_PARALELIZATION:
        with mp.Pool(processes=UserInput.CPU_CORES) as pool:
            result_list = pool.map(paralelization_process, [(name, spectral_filtered_input, UserInput.SPECTRAL_DB_PATH, 
                                                             UserInput.BAND_DISTANCE_CHECK) for name in unique_names])
    else:
        result_list: list[dict[str, float]] = []
        for name in unique_names:
            spectral_data = fetch_spectral_data(UserInput.SPECTRAL_DB_PATH, name)

            # Filtering
            spectral_filtered_database = get_spectra_filtered_list(spectral_data, UserInput.BAND_DISTANCE_CHECK)
            
            # Authoral Algorithm
            result_dict = local_algorithm(spectral_filtered_database, spectral_filtered_input, get_dataframe=False)

            result_list.append(result_dict)

    final_result = order_result_dict(result_list)
    spectral_list = [fetch_spectral_data(UserInput.SPECTRAL_DB_PATH, spectra) for spectra in final_result.keys()]

    components_data_filter_list = get_filtered_data(spectral_list, UserInput.BAND_DISTANCE_CHECK, spectral_filtered_input, get_input=False)
    input_list = get_filtered_data(spectral_list, UserInput.BAND_DISTANCE_CHECK, spectral_filtered_input, get_input=True)
    
    components_data_filter: dict[str, pd.DataFrame] = convert_to_dict(components_data_filter_list)
    input_list_dict: dict[str, pd.DataFrame] = convert_to_dict(input_list)

    print(f'Extraction and filtering: {time.perf_counter() - start_time} seconds.')

    start_time = time.perf_counter()
    
    # Report Generation
    create_graph(components_data_filter, input_list_dict, spectral_list, input_df, final_result, 
                 UserInput.ANALYSIS_COMPOUND_NAME, UserInput.OUTPUT_PDF, UserInput.BAND_DISTANCE_CHECK, UserInput.CPU_CORES)
    
    print(f'Report Generation: {time.perf_counter() - start_time} seconds.')

if __name__ == "__main__":
    main()