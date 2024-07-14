import numpy as np
import pandas as pd


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


def _get_name_and_df(spectral_df: pd.DataFrame, band_distance_check: int) -> pd.DataFrame:
    baseline_point = _define_baseline(spectral_df)
    spectral_df['baseline'] = baseline_point
    spectral_df = _calculate_band_height(spectral_df, baseline_point)
    spectral_df_band = _identify_bands_and_filter(spectral_df, band_distance_check)

    return spectral_df_band.reset_index(drop=True)


def get_spectra_filtered_list(spectra: pd.DataFrame, band_distance_check: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    spectra_output = _get_name_and_df(spectra, band_distance_check)
    return spectra_output
