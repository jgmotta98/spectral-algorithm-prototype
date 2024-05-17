import time

import numpy as np
import pandas as pd
from collections import OrderedDict
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


def local_algorithm(spectral_filtered_list: list[pd.DataFrame], spectral_input_list: list[pd.DataFrame], 
                    analysis_compound: str) -> None:
    
    spectral_filtered_list = [spectra.reset_index(drop=True) for spectra in spectral_filtered_list]
    spectral_input_list = [spectra.reset_index(drop=True) for spectra in spectral_input_list]
    
    df_dict = {v.iloc[0, 0]: v for v in spectral_filtered_list}
    df_input_dict = {v.iloc[0, 0]: v for v in spectral_input_list}

    components_data = {k: v for k, v in df_dict.items()}
    input_spectrum = [v for k, v in df_input_dict.items()][0]

    components_data_filter = {}

    similarity_scores_x = {}
    similarity_scores_height = {}
    similarity_scores_y = {}

    input_list_dict = {}

    teste_dict = {}
    for component_name, component_df in components_data.items():
        baseline_value = input_spectrum.loc[0, 'baseline']
        proportion_medium = .5 # 50%
        proportion_small = .3 # 15%

        big_band_weight = 3
        medium_band_weight = 1
        small_band_weight = 0 # descartando bandas pequenas completamente!
        
        complete_df = compare_and_filter(input_spectrum, component_df)
        
        input_df = complete_df[['input_x', 'height_input', 'input_y']]
        new_column_names = {'input_x': 'x', 'height_input': 'height', 'input_y': 'y'}
        input_df = input_df.rename(columns=new_column_names)

        new_component_df = complete_df[['component_x', 'height_component', 'component_y']]
        new_column_names = {'component_x': 'x', 'height_component': 'height', 'component_y': 'y'}
        new_component_df = new_component_df.rename(columns=new_column_names)

        def assign_band_weight(row):
            medium = proportion_medium * baseline_value
            small = proportion_small * baseline_value

            if (baseline_value - row['y']) > medium and row['y'] != 0:
                return big_band_weight
            elif (small < (baseline_value - row['y'])) and ((baseline_value - row['y']) <= medium) and row['y'] != 0:
                return medium_band_weight
            else:
                return small_band_weight
            
        def assign_x_weight(input_df, new_component_df):
            x_values = abs(input_df['x'] - new_component_df['x'])
            x_values = x_values.apply(lambda x: 3 if x > 10 else 1)
            return x_values.tolist()

        input_df['band_weight'] = input_df.apply(assign_band_weight, axis=1)
        new_component_df['band_weight'] = new_component_df.apply(assign_band_weight, axis=1)

        def regra_de_3(first_member, second_member):
            if second_member < first_member:
                return (second_member)/(first_member) if first_member and second_member else 0
            else:
                return (first_member)/(second_member) if first_member and second_member else 0
            
        # somando uma banda pequena com uma banda grande -> 0 + 3 -> quando somar uma banda pequena com qualquer outra banda. A banda pequena deve ter peso 3. (equivalente a comparação de bandas grandes!)
        # grande = 3; media = 1; pequena = 0
        # grande + grande = 6; grande + media = 4; grande + pequena = 3;
        # media + media = 2; media + pequena = 1;
        # quando for 1 ou 3 -> adicionar mais 3 -> equivalente a comparar grande com grande e média com grande!

        x_weight_sum = assign_x_weight(input_df, new_component_df)
        band_weight_sum = [(x + y) if ((x + y) != 3) and ((x + y) != 1) else (x + y + 3) for x, y in zip(input_df['band_weight'].tolist(), new_component_df['band_weight'].tolist())]
        zero_indexes = [idx for idx, val in enumerate(band_weight_sum) if val == 0]

        new_component_df = new_component_df.drop(zero_indexes).reset_index(drop=True)
        input_df = input_df.drop(zero_indexes).reset_index(drop=True)
        band_weight_sum = [weight for weight in band_weight_sum if weight != 0]
        x_weight_sum = [item for index, item in enumerate(x_weight_sum) if index not in zero_indexes]

        components_data_filter[component_name] = new_component_df
        input_list_dict[component_name] = input_df

        input_df_x = input_df['x'].tolist()
        new_component_df_x = new_component_df['x'].tolist()

        indexes_list = [i for i, num in enumerate(input_df_x) if num == 0] + [i for i, num in enumerate(new_component_df_x) if num == 0]

        input_df_x = [num for i, num in enumerate(input_df_x) if i not in indexes_list]
        new_component_df_x = [num for i, num in enumerate(new_component_df_x) if i not in indexes_list]

        correlation_coefficient_x, _ = pearsonr(input_df_x, new_component_df_x)
        correlation_coefficient_y, _ = pearsonr(input_df['y'].tolist(), new_component_df['y'].tolist())
        correlation_coefficient_height, _ = pearsonr(input_df['height'].tolist(), new_component_df['height'].tolist())
        teste_dict[component_name] = ((correlation_coefficient_x + correlation_coefficient_y + correlation_coefficient_height)/3)*100

        similarity_x = np.mean([regra_de_3(inpt, comp) for inpt, comp in zip(input_df['x'].tolist(), new_component_df['x'].tolist())])
        similarity_height = np.average([regra_de_3(inpt, comp) for inpt, comp in zip(input_df['height'].tolist(), new_component_df['height'].tolist())], 
                                       weights=band_weight_sum)
        similarity_y = np.average([regra_de_3(inpt, comp) for inpt, comp in zip(input_df['y'].tolist(), new_component_df['y'].tolist())],
                                   weights=band_weight_sum)

        similarity_percentage_x = similarity_x * 100
        similarity_scores_x[component_name] = similarity_percentage_x

        similarity_percentage_height = similarity_height * 100
        similarity_scores_height[component_name] = similarity_percentage_height

        similarity_percentage_y = similarity_y * 100
        similarity_scores_y[component_name] = similarity_percentage_y

    teste = OrderedDict(sorted(teste_dict.items(), key=lambda x: x[1], reverse=True))
    teste = dict(list(teste.items())[:5])

    print(f'Authoral Algorithm Results: {list(teste.keys())}')

    closest_matches_x = sorted(similarity_scores_x.items(), key=lambda x: x[1], reverse=True)
    closest_matches_height = sorted(similarity_scores_height.items(), key=lambda x: x[1], reverse=True)
    closest_matches_y = sorted(similarity_scores_y.items(), key=lambda x: x[1], reverse=True)

    components = {}
    for (component, similarity_percentage) in closest_matches_x:
        for (component_height, similarity_percentage_height) in closest_matches_height:
            for (component_y, similarity_percentage_y) in closest_matches_y:
                if component == component_height and component == component_y:
                    components[component] = ((similarity_percentage + similarity_percentage_height + similarity_percentage_y)/3)

    components = OrderedDict(sorted(components.items(), key=lambda x: x[1], reverse=True))
    components = dict(list(components.items())[:5])

    return components_data_filter, input_list_dict, teste
    #return components_data_filter, input_list_dict, components
        