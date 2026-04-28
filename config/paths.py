import os

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(CONFIG_DIR)
REPO_ROOT = os.path.dirname(PIPELINE_ROOT)
DATA_ROOT = os.path.join(REPO_ROOT, 'fnn', 'data', 'microns_digital_twin')

SRC_DIR = os.path.join(DATA_ROOT, 'params')
PARAMS_CSV = os.path.join(DATA_ROOT, 'params', 'units.csv')
ANATOMY_CSV = os.path.join(DATA_ROOT, 'properties', 'anatomy', 'units.csv')
PERFORMANCE_CSV = os.path.join(DATA_ROOT, 'properties', 'performance', 'units.csv')
STIM_PATH = os.path.join(REPO_ROOT, 'fnn', 'data', 'train_digital_twin', 'training_data_27203_4_7', 'stimuli')

TAESD_PATH = os.path.join(PIPELINE_ROOT, 'generation', 'taesd')

DATA_DIR = os.path.join(PIPELINE_ROOT, 'data')
FEATURES_DIR = os.path.join(PIPELINE_ROOT, 'features')
FIGURES_DIR = os.path.join(PIPELINE_ROOT, 'figures')
RESULTS_DIR = os.path.join(PIPELINE_ROOT, 'results')
RESULTS_MEDI_DIR = os.path.join(RESULTS_DIR, 'MEDI')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_MEDI_DIR, exist_ok=True)
