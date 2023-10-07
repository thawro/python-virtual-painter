from src.utils.config import DS_ROOT, NOW, ROOT

IMGSZ = 256
N_CHANNELS = 3
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_INPUT_SIZE = (IMGSZ, IMGSZ, N_CHANNELS)

EXPERIMENT_NAME = "test"
DS_PATH = DS_ROOT / "<dataset>"
SEED = 42
MODEL_INPUT_SIZE = (N_CHANNELS, IMGSZ, IMGSZ)
MODEL_INPUT_SIZE = (1, *MODEL_INPUT_SIZE)
MAX_EPOCHS = 100
BATCH_SIZE = 12
DEVICE = "cuda"


LIMIT_BATCHES = -1
LOG_EVERY_N_STEPS = 100
CKPT_PATH = "<ckpt_path>"

if LIMIT_BATCHES != -1:
    EXPERIMENT_NAME = "debug"
RUN_NAME = f"{NOW}"
LOGS_PATH = str(ROOT / "results" / EXPERIMENT_NAME / RUN_NAME)

CONFIG = {
    "seed": SEED,
    "dataset": DS_PATH,
    "model_input_size": MODEL_INPUT_SIZE,
    "max_epochs": MAX_EPOCHS,
    "batch_size": BATCH_SIZE,
    "device": DEVICE,
    "ckpt_path": CKPT_PATH,
    "limit_batches": LIMIT_BATCHES,
    "log_every_n_steps": LOG_EVERY_N_STEPS,
    "logs_path": LOGS_PATH,
    "experiment_name": EXPERIMENT_NAME,
}
