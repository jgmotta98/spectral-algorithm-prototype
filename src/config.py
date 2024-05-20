from dataclasses import dataclass
from multiprocessing import cpu_count

@dataclass
class UserInput:
    SPECTRAL_DB_PATH = '.\\files\\database\\spectral_database.db'
    BAND_DISTANCE_CHECK = 25 # Entre 10 e 40 -> Com o ideal sendo 25.
    CPU_CORES = cpu_count()
    USE_PARALELIZATION = True
    OUTPUT_PDF = '.\\files\\reports\\report_Endosulfan-sulfate.pdf'
    INPUT_PATH = '.\\files\\csv_tests\\Endosulfan-sulfate.csv'
    ANALYSIS_COMPOUND_NAME = 'Endosulfan-sulfate_teste'