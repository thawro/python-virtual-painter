"""Train the model"""

import torch
from src.data import DataModule
from src.data.transforms import ImageTransform
from src.logging import TerminalLogger, get_pylogger
from src.callbacks import (
    LoadModelCheckpoint,
    MetricsPlotterCallback,
    MetricsSaverCallback,
    ModelSummary,
    SaveModelCheckpoint,
)

from src.model.module import Trainer, BaseModule
from src.model.metrics import BaseMetrics
from src.model.utils import seed_everything

from src.utils import DS_ROOT, NOW, ROOT
from src.bin.config import (
    IMGSZ,
    MEAN,
    STD,
    SEED,
    DS_PATH,
    BATCH_SIZE,
    MODEL_INPUT_SIZE,
    CKPT_PATH,
    LOGS_PATH,
    LOG_EVERY_N_STEPS,
    CONFIG,
    DEVICE,
    MAX_EPOCHS,
    LIMIT_BATCHES,
)

log = get_pylogger(__name__)

transform = ImageTransform(IMGSZ, MEAN, STD)


def create_datamodule() -> DataModule:
    train_ds = None
    val_ds = None
    test_ds = None
    return DataModule(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, batch_size=BATCH_SIZE)

def create_module() -> BaseModule:
    loss_fn = None
    model = None
    optimizer None
    scheduler None
    metrics = BaseMetrics()
    module = BaseModule(
        model=model,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizers={"optim": optimizer},
        schedulers={"optim": scheduler},
    )
    return module


def create_callbacks(logger: TerminalLogger) -> list:
    ckpt_saver_params = dict(ckpt_dir=logger.ckpt_dir, stage="val", mode="min")
    summary_filepath = str(logger.model_dir / "model_summary.txt")
    examples_dirpath = logger.log_path / "steps_examples"
    examples_dirpath.mkdir()
    callbacks = [
        MetricsPlotterCallback(str(logger.log_path / "epoch_metrics.jpg")),
        MetricsSaverCallback(str(logger.log_path / "epoch_metrics.yaml")),
        ModelSummary(input_size=BATCHED_INPUT_SIZE, depth=4, filepath=summary_filepath),
        SaveModelCheckpoint(name="best", metric="mean_IoU", **ckpt_saver_params),
        SaveModelCheckpoint(name="last", last=True, top_k=0, **ckpt_saver_params),
    ]
    if CKPT_PATH is not None:
        callbacks.append(LoadModelCheckpoint(CKPT_PATH))
    return callbacks


def main() -> None:
    seed_everything(SEED)
    torch.set_float32_matmul_precision("medium")

    datamodule = create_datamodule()
    module = create_module()

    logger = TerminalLogger(LOGS_PATH, config=CONFIG)
    callbacks = create_callbacks(logger)

    trainer = Trainer(
        logger=logger,
        device=DEVICE,
        callbacks=callbacks,
        max_epochs=MAX_EPOCHS,
        limit_batches=LIMIT_BATCHES,
        log_every_n_steps=LOG_EVERY_N_STEPS,
    )
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
