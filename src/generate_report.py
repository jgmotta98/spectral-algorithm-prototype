import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from matplotlib.figure import Figure
import os
from fpdf import FPDF


TEMP_PATH = './files/temp'


def create_graph(components_data_filter, input_list_dict, spectral_list, components, compound, output):
    dict_teste = {v.iloc[0, 0]: v.drop('name', axis=1) for v in spectral_list}

    sorted_data = dict(sorted(components.items(), key=lambda item: item[1], reverse=True))
    components = list(sorted_data.keys())

    component_dict = {comp: components_data_filter[comp].loc[:, ('x', 'y')] for comp in (components)}

    table_dict: dict[str, list[tuple[str, str, str, str, str]]] = {}
    
    for k, v in component_dict.items():
        table_list: list[tuple[str, str, str, str, str]] = []
        fig, ax = plt.subplots()
        
        input_list = [input_list_dict[k]['x'].tolist(), input_list_dict[k]['y'].tolist()]
        database_list = [v['x'].tolist(), v['y'].tolist()]

        input_zero_indexes = [i for i, value in enumerate(input_list[0]) if value == 0]
        database_zero_indexes = [i for i, value in enumerate(database_list[0]) if value == 0]

        merged_zero_indexes = list(set(input_zero_indexes + database_zero_indexes))

        input_list = [[value for idx, value in enumerate(sublist) if idx not in merged_zero_indexes] for sublist in input_list]
        database_list = [[value for idx, value in enumerate(sublist) if idx not in merged_zero_indexes] for sublist in database_list]

        number_range = len(database_list[0])

        ax.scatter(database_list[0], database_list[1], label='Database Points', zorder=2, color='blue')
        ax.scatter(input_list[0], input_list[1], label='Analysed File Points', zorder=2, color='orange')

        for i in range(number_range, 0, -1):
            table_list.append((str(i), f'{database_list[0][number_range - i]:.2f}', f'{database_list[1][number_range - i]:.2f}', 
                             f'{input_list[0][number_range - i]:.2f}', f'{input_list[1][number_range - i]:.2f}'))
            
            text = ax.annotate(i, (database_list[0][number_range - i], database_list[1][number_range - i]), zorder=4, color='blue', fontsize=10, ha='left', va='center')
            text.set_path_effects([withStroke(linewidth=3, foreground='white')])

            text = ax.annotate(i, (input_list[0][number_range - i], input_list[1][number_range - i]), zorder=4, color='orange', fontsize=10, ha='left', va='center')
            text.set_path_effects([withStroke(linewidth=3, foreground='white')])

        for i in range(len(database_list[0])):
            ax.plot([database_list[0][i], input_list[0][i]], [database_list[1][i], input_list[1][i]], linestyle='--', color='purple', alpha=0.5, zorder=3)

        ax.plot(dict_teste[compound]['x'], dict_teste[compound]['y'], color='black', zorder=1)
        ax.plot(dict_teste[k]['x'], dict_teste[k]['y'], color='red', zorder=1, alpha =0.5)
        ax.legend()
        plt.xlim((4000, 400))
        plt.ylim((0, 100))

        create_temp_png(os.path.join(TEMP_PATH, f"{k}.png"), fig)
        table_list = sorted(table_list, key=lambda x: float(x[0]))
        table_dict[k] = table_list

    create_pdf_page(TEMP_PATH, output, components, table_dict)


def create_temp_png(output_path: str, fig: Figure) -> None:
    tmp_file = os.path.join(output_path)
    fig.savefig(tmp_file)
    plt.close(fig)


def get_temp_imgs(imgs_path: str) -> list[str]:
    return [os.path.join(imgs_path, img_name) for img_name in os.listdir(imgs_path) if img_name.endswith('.png')]


def footer(pdf, tmp_file):
        pdf.set_y(-15)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, os.path.basename(tmp_file).split('.')[0], 0, 0, 'C')


def create_table(pdf: FPDF, table_data: list[tuple[str, str, str, str, str]], tmp_file: str) -> None:
    table_header = [('Id', 'Wavelength (cm^1)', 'Intensity', 'Wavelength (cm^1)', 'Intensity')]
    table_data = table_header + table_data

    pdf.add_page()
    pdf.set_font("Times", size=12)
    with pdf.table(text_align="CENTER", borders_layout="SINGLE_TOP_LINE") as table:
        for data_row in table_data:
            row = table.row()
            for datum in data_row:
                row.cell(datum)

    pdf.footer = lambda: footer(pdf, tmp_file)


def create_pdf_page(imgs_path: str, output_pdf: str, 
                    compound_list: list[str], table_dict: dict[str, list[tuple[str, str, str, str, str]]]) -> None:

    temp_img_list = get_temp_imgs(imgs_path)

    file_dict = {os.path.basename(filename).split('.')[0]: filename for filename in temp_img_list}

    temp_img_list = [file_dict[key] for key in compound_list]
    
    pdf = FPDF(orientation="landscape")
    
    for tmp_file in temp_img_list:
        pdf.add_page()
        pdf.footer = lambda: footer(pdf, tmp_file)
        pdf.image(tmp_file)
        create_table(pdf, table_dict[os.path.basename(tmp_file).split('.')[0]], tmp_file)
    pdf.output(output_pdf)

    [os.remove(tmp_file) for tmp_file in temp_img_list]

    for i in range(1, pdf.page_no() + 1):
        pdf.in_footer = lambda pdf: custom_footer(pdf, i)

def custom_footer(pdf, page_num):
    if page_num == 2:
        pdf.set_y(-15)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, f"Custom Footer for Page {page_num}", align="C")
    else:
        pdf.set_y(-15)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, f"Footer for Page {page_num}", align="C")