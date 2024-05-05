from multiprocessing import cpu_count

from band_filter import get_spectra_filtered_list
from similarity_algorithm import local_algorithm
from generate_report import create_graph


SPECTRAL_DB_PATH = r'D:\Downloads\Material Faculdade\Material TCC\spectral-algorithm-prototype\files\database\spectra_data.db'
BAND_DISTANCE_CHECK = 25
CPU_CORES = cpu_count()
OUTPUT_PDF = r'D:\Downloads\Material Faculdade\Material TCC\spectral-algorithm-prototype\files\reports\output_horizontal_centralized.pdf'
ANALYSIS_COMPOUND = 'trans-1,3-pentadiene'


def main() -> None:
    spectral_filtered_list, spectral_list = get_spectra_filtered_list(SPECTRAL_DB_PATH, BAND_DISTANCE_CHECK, cpu_cores=CPU_CORES)
    components_data_filter, input_list_dict, components_result = local_algorithm(spectral_filtered_list, ANALYSIS_COMPOUND)
    create_graph(components_data_filter, input_list_dict, spectral_list, components_result, ANALYSIS_COMPOUND, OUTPUT_PDF, BAND_DISTANCE_CHECK)


if __name__ == "__main__":
    main()