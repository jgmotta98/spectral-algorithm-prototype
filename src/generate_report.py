from reportlab.lib.pagesizes import landscape
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageTemplate, Frame, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from PyPDF2 import PdfMerger
import os
import re

TEMP_PATH = './files/temp'


def create_graph(components_data_filter, input_list_dict, spectral_list, components, compound, output):
    dict_teste = {v.iloc[0, 0]: v.drop('name', axis=1) for v in spectral_list}

    sorted_data = dict(sorted(components.items(), key=lambda item: item[1], reverse=True))
    components = list(sorted_data.keys())

    component_dict = {comp: components_data_filter[comp].loc[:, ('x', 'y')] for comp in (components)}

    for k, v in component_dict.items():
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
            text = ax.annotate(i, (database_list[0][number_range - i], database_list[1][number_range - i]), zorder=4, color='blue', fontsize=10, ha='left', va='center')
            text.set_path_effects([withStroke(linewidth=3, foreground='white')])

            text = ax.annotate(i, (input_list[0][number_range - i], input_list[1][number_range - i]), zorder=4, color='orange', fontsize=10, ha='left', va='center')
            text.set_path_effects([withStroke(linewidth=3, foreground='white')])

        for i in range(len(database_list[0])):
            ax.plot([database_list[0][i], input_list[0][i]], [database_list[1][i], input_list[1][i]], linestyle='--', color='purple', alpha=0.5, zorder=3)

        ax.plot(dict_teste[compound]['x'], dict_teste[compound]['y'], color='black', zorder=1)
        ax.plot(dict_teste[k]['x'], dict_teste[k]['y'], color='red', zorder=1, alpha =0.5)
        plt.title(k)
        ax.legend()
        plt.xlim((4000, 400))
        plt.ylim((0, 100))
        figure = plt.gcf()
        figure.set_size_inches(32, 18)

        create_pdf_page(os.path.join(TEMP_PATH, f"{k}_img.pdf"), fig)
        teste(os.path.join(TEMP_PATH, f"{k}_table.pdf"))
        merge_temp_pdfs(TEMP_PATH)
    merge_pdfs(TEMP_PATH, output, components) 


def create_pdf_page(output_pdf, fig):
    buffer = BytesIO()

    c = canvas.Canvas(buffer, pagesize=landscape(letter))

    tmp_file = os.path.join(TEMP_PATH, 'temp_plot.png')
    fig.savefig(tmp_file, dpi=150)
    plt.close(fig)

    page_width, page_height = landscape(letter)
    graph_width, graph_height = 800, 600
    x_offset = (page_width - graph_width) / 2
    y_offset = (page_height - graph_height) / 2

    c.drawInlineImage(tmp_file, x_offset, y_offset, width=graph_width, height=graph_height)

    c.showPage()
    c.save()

    with open(output_pdf, 'ab') as f:
        f.write(buffer.getvalue())

    buffer.close()
    os.remove(tmp_file)


def get_pdf_temp(temp_path: str) -> list[str]:
    return [os.path.join(temp_path, files) for files in os.listdir(temp_path) if files.endswith('.pdf')]


def get_img_table_pdf(temp_path: list[str]) -> list[str]:
    return [files for files in temp_path if files.split("_")[-1] in ['img.pdf', 'table.pdf']]


def merge_pdfs(input_path, output_pdf, compound_list):
    temp_pdf_list = get_pdf_temp(input_path)
    merger = PdfMerger()

    file_dict = {os.path.basename(filename).split('.')[0]: filename for filename in temp_pdf_list}

    temp_pdf_list = [file_dict[key] for key in compound_list]

    for pdf_file in  temp_pdf_list:
        merger.append(pdf_file)

    merger.write(output_pdf)

    merger.close()
    [os.remove(pdf_file) for pdf_file in temp_pdf_list]


def merge_temp_pdfs(input_path):
    temp_pdf_list = get_pdf_temp(input_path)
    temp_pdf_list = get_img_table_pdf(temp_pdf_list)
    pattern = re.compile(r'(_img|_table)')
    file_name = os.path.basename(re.sub(pattern, '', temp_pdf_list[0]))

    merger = PdfMerger()
    
    for pdf_file in  temp_pdf_list:
        merger.append(pdf_file)

    merger.write(os.path.join(input_path, file_name))

    merger.close()
    [os.remove(pdf_file) for pdf_file in temp_pdf_list]


def teste(file_path: str) -> None:
    doc = SimpleDocTemplate(file_path, pagesize=landscape(letter))

    data = [
        ["Name", "Age", "Country"],
        ["John Doe", "30", "USA"],
        ["Jane Smith", "25", "UK"],
        ["Ahmed Khan", "35", "India"],
    ]

    table = Table(data)

    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey), 
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), 
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])
    table.setStyle(style)

    footer_text = "This is a footer note."

    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.drawString(30, 20, footer_text)
        canvas.restoreState()

    styles = getSampleStyleSheet()
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
    footer_frame = Frame(doc.leftMargin, doc.bottomMargin - 50, doc.width, 50, id='footer')
    footer_template = PageTemplate(id='footer', frames=[frame, footer_frame], onPage=add_footer)
    doc.addPageTemplates([footer_template])

    elements = []
    elements.append(table)
    doc.build(elements)