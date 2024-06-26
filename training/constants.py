from pathlib import Path

HYPERS_DIR = Path(__file__).parent.parent / 'hypers'
TORCH_SEED = 42
PROJECT_NAME = 'MGT-local'
DATA_AUG_PARAMS_FILENAME = "dataAugParams.json"
HIT_SIGMOID_IN_FORWARD = False
LOG_EVERY = 256
VOICES = 9
TIME_STEPS = 32
MAX_LEN = TIME_STEPS