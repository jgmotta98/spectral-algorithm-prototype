import time
from multiprocessing import Pool, cpu_count
import sqlite3

import numpy as np
import pandas as pd


def _retrieve_data(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT SpectraNames.name, SpectraData.wavelength, SpectraData.intensity 
        FROM SpectraNames 
        INNER JOIN SpectraData ON SpectraNames.id = SpectraData.name_id
    ''')
    data = cursor.fetchall()

    df = pd.DataFrame(data, columns=['name', 'wavelength', 'intensity'])

    conn.close()
    return df


def _get_database_values(db_path: str) -> dict[str, pd.DataFrame]:
    df = _retrieve_data(db_path)
    #dfs = {name: group.drop('name', axis=1).reset_index(drop=True) for name, group in df.groupby('name')}
    dfs = [group.reset_index(drop=True).rename(columns={'wavelength': 'x', 'intensity': 'y'}) for _, group in df.groupby('name')]
    return dfs


def _define_baseline(spectral_df: pd.DataFrame) -> float:
    baseline_idx = spectral_df['y'].idxmax()
    baseline_point = spectral_df.loc[baseline_idx, 'y']

    return baseline_point


def _calculate_band_height(spectral_df: pd.DataFrame,
                          baseline_point: float) -> pd.DataFrame:
    spectral_df['height'] = baseline_point - spectral_df['y']
    
    return spectral_df


def _filtering_operations(x_values_list: list[float], window_size: int, 
                          heights: list[float]) -> list[int]:
    x_values_arr = np.array(x_values_list)
    heights_arr = np.array(heights)
    n = len(x_values_arr)
    result_indices: list[int] = []
    lower_bounds = x_values_arr - window_size
    upper_bounds = x_values_arr + window_size
    for i in range(n):
        indices_to_compare = np.where((x_values_arr >= lower_bounds[i]) & (x_values_arr <= upper_bounds[i]))[0]
        comparison_height = heights_arr[indices_to_compare].max()
        if heights_arr[i] >= comparison_height:
            result_indices.append(i)
    return result_indices


def _identify_bands_and_filter(spectral_df: pd.DataFrame, band_distance_check: int) -> pd.DataFrame:
    spectral_df_band = spectral_df.loc[spectral_df.groupby('x')['height'].idxmax()].reset_index(drop=True)

    x_val_list = spectral_df_band['x'].values
    heights = spectral_df_band['height'].values
    result = _filtering_operations(x_val_list, band_distance_check, heights)
    spectral_df_band = spectral_df_band.iloc[result].drop_duplicates(subset='height', keep='first')

    return spectral_df_band


def _get_name_and_df(spectral_list: list[pd.DataFrame], band_distance_check: int) -> list[pd.DataFrame]:
    spectral_filtered_list: list[pd.DataFrame] = []

    for spectral_df in spectral_list:
        baseline_point = _define_baseline(spectral_df)
        spectral_df = _calculate_band_height(spectral_df, baseline_point)
        spectral_df_band = _identify_bands_and_filter(spectral_df, band_distance_check)
        spectral_filtered_list.append(spectral_df_band)

    return spectral_filtered_list


def _process_chunk(args: list[tuple[pd.DataFrame, ...], int]) -> list[pd.DataFrame]:
    chunk, band_distance_check = args
    return _get_name_and_df(chunk, band_distance_check)


def get_spectra_filtered_list(db_path: str, band_distance_check: int, *, cpu_cores: int = cpu_count()) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    start_time = time.perf_counter()
    spectral_list = _get_database_values(db_path)

    num_cores = cpu_cores
    chunk_size = len(spectral_list) // num_cores

    spectral_list_chunks = [spectral_list[i:i + chunk_size] for i in range(0, len(spectral_list), chunk_size)]

    args = [(chunk, band_distance_check) for chunk in spectral_list_chunks]

    with Pool(num_cores) as pool:
        processed_chunks = pool.map(_process_chunk, args)

    spectral_filtered_list = [item for sublist in processed_chunks for item in sublist]
    print(f'Execution time: {time.perf_counter() - start_time} seconds.')
    return spectral_filtered_list, spectral_list