from band_filter import get_spectra_filtered_list
from similarity_algorithm import local_algorithm
from multiprocessing import cpu_count

SPECTRAL_DB_PATH = './database/spectra_data.db'
BAND_DISTANCE_CHECK = 25
CPU_CORES = cpu_count()

def main() -> None:
    spectral_filtered_list, spectral_list = get_spectra_filtered_list(SPECTRAL_DB_PATH, BAND_DISTANCE_CHECK, cpu_cores=CPU_CORES)
    local_algorithm(spectral_filtered_list, 'isopentane')

if __name__ == "__main__":
    main()