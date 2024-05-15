import time
from multiprocessing import Pool, cpu_count
import sqlite3

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, argrelmin
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve


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
        spectral_df['baseline'] = baseline_point
        spectral_df = _calculate_band_height(spectral_df, baseline_point)
        spectral_df_band = _identify_bands_and_filter(spectral_df, band_distance_check)
        spectral_filtered_list.append(spectral_df_band)

    return spectral_filtered_list


def _process_chunk(args: list[tuple[pd.DataFrame, ...], int]) -> list[pd.DataFrame]:
    chunk, band_distance_check = args
    return _get_name_and_df(chunk, band_distance_check)


def create_compound_line(spectral_df_list: list[pd.DataFrame], range_value: int) -> pd.DataFrame:
    wavelength = np.arange(400, 4001, range_value)

    wavelength = [f"{wavelength[i]} - {wavelength[i + 1]}" for i in range(len(wavelength) - 1)] + [f"{wavelength [-1]} - {int(wavelength[-1]) + range_value}"][:-1]
    column = np.array(['Name'])
    column = np.concatenate((column, wavelength), axis=0)

    df = pd.DataFrame()

    for df_value in spectral_df_list:
        data: dict[str, float] = {}

        wavelength_values = df_value['x'].tolist()
        intensity_values = df_value['height'].tolist()

        data['Name'] = [df_value.loc[0, 'name']]
        for k in wavelength:
            data[k] = [0]

        for k in data.keys():
            for k2, v2 in zip(wavelength_values, intensity_values):
                if k != 'Name':
                    if int(k.split("-")[0]) < k2 < int(k.split("-")[1]) and data[k][0] < v2:
                        data[k] = [v2]

        new_df = pd.DataFrame(data)

        if df.empty:
            df = new_df
        else:
            df = pd.concat([df, new_df], axis=0)

    df.reset_index(inplace=True, drop=True)
    return df


def round_and_filter(spectral_df: pd.DataFrame) -> pd.DataFrame:
    spectral_df['x'] = spectral_df['x'].astype(int)
    sorted_df = spectral_df.sort_values(by=['name', 'height'], ascending=[True, False])
    spectral_df = sorted_df.groupby('x').first().reset_index()

    return spectral_df


def round_and_filter_pre(spectral_df: pd.DataFrame) -> pd.DataFrame:
    spectral_df['x'] = spectral_df['x'].astype(int)
    sorted_df = spectral_df.sort_values(by=['name', 'y'], ascending=[True, True])
    filtered_df = sorted_df.loc[sorted_df.groupby('x')['y'].idxmin()].reset_index(drop=True)
    filtered_df['y'] = filtered_df['y'].apply(lambda x: 100 if x > 100 else x)

    return filtered_df


# ------- Experimental ----------------

def WhittakerSmooth(x: list[float], w: np.ndarray[float], 
                    lambda_: int, differences: int = 1) -> np.ndarray:
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    E=eye(m,format='csc')
    for i in range(differences):
        E=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*E.T*E))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)


def airPLS(x: list[float], lambda_: int = 100, 
           porder: int = 1, itermax: int = 15) -> np.ndarray:
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARNING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z


#---------------------------------------

def get_spectra_filtered_list(db_path: str, band_distance_check: int, input_df: list[str, pd.DataFrame], *, cpu_cores: int = cpu_count()) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    start_time = time.perf_counter()
    spectral_list = _get_database_values(db_path)
    
    spectral_list = [round_and_filter_pre(spectra) for spectra in spectral_list]

    input_df[1].insert(0, 'name', input_df[0])
    input_df = input_df[1]
    
    input_df['y'] = 2 - np.log10(input_df['y'])
    c1=np.array(input_df['y'].tolist())-airPLS(np.array(input_df['y'].tolist())) # adicionar a possibilidade do usuario de alterar a porder!
    
    c1 = 10**(-c1) * 100
    input_df['y'] = c1.tolist()

    input_df = round_and_filter_pre(input_df)
    
    num_cores = cpu_cores
    chunk_size = len(spectral_list) // num_cores

    spectral_list_chunks = [spectral_list[i:i + chunk_size] for i in range(0, len(spectral_list), chunk_size)]

    args = [(chunk, band_distance_check) for chunk in spectral_list_chunks]

    with Pool(num_cores) as pool:
        processed_chunks = pool.map(_process_chunk, args)

    spectral_input_list = _process_chunk(([input_df], band_distance_check))

    spectral_filtered_list = [item for sublist in processed_chunks for item in sublist]

    print(f'Execution time: {time.perf_counter() - start_time} seconds.')

    spectral_alt_df_filtered_list = [round_and_filter(df) for df in spectral_filtered_list]

    spectral_alt_df_filtered_list.append(round_and_filter(spectral_input_list[0]))

    spectral_df = create_compound_line(spectral_alt_df_filtered_list, band_distance_check)

    spectral_df.to_pickle(r'D:\Downloads\Material Faculdade\Material TCC\spectral-algorithm-prototype\files\database\teste.pickle')

    return spectral_filtered_list, spectral_input_list, spectral_list