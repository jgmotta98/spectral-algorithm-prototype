import time

import numpy as np
import pandas as pd


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


def local_algorithm(spectral_filtered_list: list[pd.DataFrame], analysis_compound: str) -> None:
    
    df_dict = {v.iloc[0, 0]: v for v in spectral_filtered_list}
    components_data = {k: v for k, v in df_dict.items() if k != analysis_compound}
    input_spectrum = [v for k, v in df_dict.items() if k == analysis_compound][0]

    components_data_filter = {}

    similarity_scores_x = {}
    similarity_scores_height = {}
    similarity_scores_y = {}

    input_list_dict = {}
    start_time = time.perf_counter()
    for component_name, component_df in components_data.items():
        complete_df = compare_and_filter(input_spectrum, component_df)
        
        input_df = complete_df[['input_x', 'height_input', 'input_y']]
        new_column_names = {'input_x': 'x', 'height_input': 'height', 'input_y': 'y'}
        input_df = input_df.rename(columns=new_column_names)

        input_list_dict[component_name] = input_df

        new_component_df = complete_df[['component_x', 'height_component', 'component_y']]
        new_column_names = {'component_x': 'x', 'height_component': 'height', 'component_y': 'y'}
        new_component_df = new_component_df.rename(columns=new_column_names)

        components_data_filter[component_name] = new_component_df

        def regra_de_3(first_member, second_member):
            if second_member < first_member:
                return (second_member)/(first_member)
            else:
                return (first_member)/(second_member) if first_member and second_member else 0
            
        similarity_x = np.mean([regra_de_3(inpt, comp) for inpt, comp in zip(input_df['x'].tolist(), new_component_df['x'].tolist())])
        similarity_height = np.mean([regra_de_3(inpt, comp) for inpt, comp in zip(input_df['height'].tolist(), new_component_df['height'].tolist())])
        similarity_y = np.mean([regra_de_3(inpt, comp) for inpt, comp in zip(input_df['y'].tolist(), new_component_df['y'].tolist())])

        similarity_percentage_x = similarity_x * 100
        similarity_scores_x[component_name] = similarity_percentage_x

        similarity_percentage_height = similarity_height * 100
        similarity_scores_height[component_name] = similarity_percentage_height

        similarity_percentage_y = similarity_y * 100
        similarity_scores_y[component_name] = similarity_percentage_y
    print(f'Execution time: {time.perf_counter() - start_time} seconds.')

    closest_matches_x = sorted(similarity_scores_x.items(), key=lambda x: x[1], reverse=True)
    closest_matches_height = sorted(similarity_scores_height.items(), key=lambda x: x[1], reverse=True)
    closest_matches_y = sorted(similarity_scores_y.items(), key=lambda x: x[1], reverse=True)
    print('Component real name: ', analysis_compound)
    components = {}
    for index, (component, similarity_percentage) in enumerate(closest_matches_x):
        if index <= 4:
            for (component_height, similarity_percentage_height) in closest_matches_height:
                for (component_y, similarity_percentage_y) in closest_matches_y:
                    if component == component_height and component == component_y:
                        components[component] = ((similarity_percentage + similarity_percentage_height + similarity_percentage_y)/3)
                        print(f"{component}: Similarity by X = {similarity_percentage:.2f}% ///// Similarity by Height = {similarity_percentage_height:.2f}% //// Similarity by Intensity = {similarity_percentage_y:.2f}%")

    return components_data_filter, input_list_dict, components