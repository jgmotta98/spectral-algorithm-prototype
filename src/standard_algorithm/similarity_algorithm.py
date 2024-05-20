import pandas as pd
from scipy.stats import pearsonr


def compare_and_filter(input_spectrum: pd.DataFrame, component_df: pd.DataFrame) -> pd.DataFrame:
    spectrum_x = input_spectrum['x'].tolist()
    component_x = component_df['x'].tolist()

    input_values = spectrum_x
    component_values = component_x

    pairs = []
    
    used_components = []
    for idx, input_val in enumerate(input_values):
        min_diff = float('inf')
        best_component = None

        for component_val in component_values:
            diff = abs(input_val - component_val)
            if diff < min_diff:
                if component_val not in used_components:
                    if not used_components or (used_components[-1] is not None and component_val >= used_components[-1]):
                        if idx + 1 < len(input_values) and abs(input_val - component_val) < abs(input_values[idx + 1] - component_val):
                            min_diff = diff
                            best_component = component_val
                        elif idx + 1 >= len(input_values):
                            min_diff = diff
                            best_component = component_val

        if best_component:
            used_components.append(best_component)

        pairs.append((input_val, best_component))

    for val in component_values:
        found = False
        for item in pairs:
            if item[1] == val:
                found = True
                break
        if not found:
            pairs.append((0, val))
    
    #pairs = [v for v in pairs if v[1] is not None]

    input_x_list = []
    component_x_list = []
    height_input_list = []
    height_component_list = []
    input_y_list = []
    component_y_list = []

    for input_x, component_x in pairs:
        if input_x != 0:
            input_x_list.append(input_x)
            height_input_list.append(input_spectrum[input_spectrum['x'] == input_x]['height'].values[0])
            input_y_list.append(input_spectrum[input_spectrum['x'] == input_x]['y'].values[0])
        else:
            input_x_list.append(0)
            height_input_list.append(0)
            input_y_list.append(0)

        if component_x is not None:
            component_x_list.append(component_x)
            height_component_list.append(component_df[component_df['x'] == component_x]['height'].values[0])
            component_y_list.append(component_df[component_df['x'] == component_x]['y'].values[0])
        else:
            component_x_list.append(0)
            height_component_list.append(0)
            component_y_list.append(0)
    
    new_df = pd.DataFrame({
        'input_x': input_x_list,
        'component_x': component_x_list,
        'height_input': height_input_list,
        'height_component': height_component_list,
        'input_y': input_y_list,
        'component_y': component_y_list
    })

    return new_df


def assign_band_weight(row: pd.Series, upper_portion: float, lower_portion: float, baseline_value: float,
                       big_band_weight: int, medium_band_weight: int, small_band_weight: int) -> int:
        medium = upper_portion * baseline_value
        small = lower_portion * baseline_value

        if (baseline_value - row['y']) > medium and row['y'] != 0:
            return big_band_weight
        elif (small < (baseline_value - row['y'])) and ((baseline_value - row['y']) <= medium) and row['y'] != 0:
            return medium_band_weight
        else:
            return small_band_weight
        

def assign_x_weight(input_df: pd.DataFrame, new_component_df: pd.DataFrame) -> list[int]:
    x_values = abs(input_df['x'] - new_component_df['x'])
    x_values = x_values.apply(lambda x: 3 if x > 10 else 1)
    return x_values.tolist()


def local_algorithm(spectral_filtered_database: pd.DataFrame, 
                    spectral_filtered_input: pd.DataFrame, *, get_dataframe: bool = False, get_input: bool = False) -> dict[str, float] | dict[str, pd.DataFrame]:
    result_dict: dict[str, float] = {}

    components_data_filter: dict[str, pd.DataFrame] = {}
    input_list_dict: dict[str, pd.DataFrame] = {}

    baseline_value = spectral_filtered_input.loc[0, 'baseline']
    upper_portion = .5 # acima de 50% do tamanho da baseline é considerado banda grande.
    lower_portion = .3 # abaixo de 15% do tamanho da baseline é considerado banda pequena.

    big_band_weight = 2
    medium_band_weight = 1
    small_band_weight = 0 # descartando bandas pequenas completamente!
    
    complete_df = compare_and_filter(spectral_filtered_input, spectral_filtered_database)
    
    input_df = complete_df[['input_x', 'height_input', 'input_y']]
    new_column_names = {'input_x': 'x', 'height_input': 'height', 'input_y': 'y'}
    input_df = input_df.rename(columns=new_column_names)

    new_component_df = complete_df[['component_x', 'height_component', 'component_y']]
    new_column_names = {'component_x': 'x', 'height_component': 'height', 'component_y': 'y'}
    new_component_df = new_component_df.rename(columns=new_column_names)

    input_df['band_weight'] = input_df.apply(assign_band_weight, axis=1, args=(upper_portion, lower_portion, baseline_value, 
                                                                               big_band_weight, medium_band_weight, small_band_weight))
    new_component_df['band_weight'] = new_component_df.apply(assign_band_weight, axis=1, args=(upper_portion, lower_portion, baseline_value, 
                                                                                               big_band_weight, medium_band_weight, small_band_weight))

    x_weight_sum = assign_x_weight(input_df, new_component_df)
    band_weight_sum = [(x + y) for x, y in zip(input_df['band_weight'].tolist(), new_component_df['band_weight'].tolist())]
    zero_indexes = [idx for idx, val in enumerate(band_weight_sum) if val == 0]

    new_component_df = new_component_df.drop(zero_indexes).reset_index(drop=True)
    input_df = input_df.drop(zero_indexes).reset_index(drop=True)
    band_weight_sum = [weight for weight in band_weight_sum if weight != 0]
    x_weight_sum = [item for index, item in enumerate(x_weight_sum) if index not in zero_indexes]

    components_data_filter[spectral_filtered_database.loc[0, 'name']] = new_component_df
    input_list_dict[spectral_filtered_database.loc[0, 'name']] = input_df

    input_df_x = input_df['x'].tolist()
    new_component_df_x = new_component_df['x'].tolist()

    indexes_list = [i for i, num in enumerate(input_df_x) if num == 0] + [i for i, num in enumerate(new_component_df_x) if num == 0]

    input_df_x = [num for i, num in enumerate(input_df_x) if i not in indexes_list]
    new_component_df_x = [num for i, num in enumerate(new_component_df_x) if i not in indexes_list]

    correlation_coefficient_x, _ = pearsonr(input_df_x, new_component_df_x)
    correlation_coefficient_y, _ = pearsonr(input_df['y'].tolist(), new_component_df['y'].tolist())
    correlation_coefficient_height, _ = pearsonr(input_df['height'].tolist(), new_component_df['height'].tolist())
    result_dict[spectral_filtered_database.loc[0, 'name']] = ((correlation_coefficient_x + correlation_coefficient_y + correlation_coefficient_height)/3)*100

    if not get_dataframe:
        return result_dict

    if not get_input:
        return components_data_filter
    
    return input_list_dict
