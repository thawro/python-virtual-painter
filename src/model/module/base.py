from abc import abstractmethod

import torch
from torch import nn, Tensor

from src.data.datamodule import DataModule
from src.logging import get_pylogger
from src.logging.loggers import BaseLogger
from src.metrics import MetricsStorage, Result
from src.model.model import BaseModel
from torch.nn.modules.loss import _Loss
from src.model.metrics.base import BaseMetrics

from src.callbacks import Callbacks

log = get_pylogger(__name__)

SPLITS = ["train", "val", "test"]


class BaseModule:
    logger: BaseLogger
    device: torch.device
    datamodule: DataModule
    callbacks: "Callbacks"
    current_epoch: int
    current_step: int
    log_every_n_steps: int
    limit_batches: int

    def __init__(
        self,
        model: BaseModel,
        loss_fn: _Loss,
        metrics: BaseMetrics,
        optimizers: dict[str, torch.optim.Optimizer],
        schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {},
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.metrics = metrics
        self.steps_metrics_storage = MetricsStorage()
        self.epochs_metrics_storage = MetricsStorage()
        self.results: dict[str, Result] = {}

    def pass_attributes(
        self,
        device: torch.device,
        logger: BaseLogger,
        callbacks: "Callbacks",
        datamodule: DataModule,
        limit_batches: int,
        log_every_n_steps: int,
    ):
        self.logger = logger
        self.callbacks = callbacks
        self.datamodule = datamodule
        self.limit_batches = limit_batches
        self.total_batches = datamodule.total_batches
        if limit_batches > 0:
            self.total_batches = {k: limit_batches for k in self.total_batches}
        self.device = device
        self.log_every_n_steps = log_every_n_steps

    def set_attributes(self, **attributes):
        for name, attr in attributes.items():
            setattr(self, name, attr)

    def load_state_dict(self, state_dict: dict):
        self.model.load_state_dict(state_dict["model"])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state_dict["optimizers"][name])
        for name, scheduler in self.schedulers.items():
            scheduler.load_state_dict(state_dict["schedulers"][name])
        self.steps_metrics_storage.load_state_dict(state_dict["metrics"]["steps"])
        self.epochs_metrics_storage.load_state_dict(state_dict["metrics"]["epochs"])

    def state_dict(self) -> dict:
        optimizers_state = {
            name: optimizer.state_dict() for name, optimizer in self.optimizers.items()
        }
        schedulers_state = {
            name: scheduler.state_dict() for name, scheduler in self.schedulers.items()
        }
        metrics_state = {
            "steps": self.steps_metrics_storage.state_dict(),
            "epochs": self.epochs_metrics_storage.state_dict(),
        }
        model_state = {"model": self.model.state_dict()}
        model_state.update(
            {
                "optimizers": optimizers_state,
                "schedulers": schedulers_state,
                "metrics": metrics_state,
            }
        )
        return model_state

    def _common_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, stage: str, update_metrics: bool
    ):
        if stage == "train":
            self.optimizers["optim"].zero_grad()

        data, targets = batch
        preds = self.model(data)
        loss = self.loss_fn(preds, targets)

        if stage == "train":
            loss.backward()
            self.optimizers["optim"].step()

        if update_metrics:
            losses = {"loss": loss.item()}
            metrics = self.metrics.calculate_metrics(preds, targets)

            self.steps_metrics_storage.append(losses, stage)
            self.steps_metrics_storage.append(metrics, stage)

        if self.current_step % self.log_every_n_steps == 0 and batch_idx == 0:
            self.results[stage] = Result(
                data=data.detach().cpu().numpy(),
                preds=preds.detach().cpu().numpy(),
                targets=targets.cpu().numpy(),
            )

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, update_metrics: bool = True
    ):
        self._common_step(batch, batch_idx, "train", update_metrics)

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int, update_metrics: bool):
        self._common_step(batch, batch_idx, "val", update_metrics)

    @abstractmethod
    def on_train_epoch_start(self):
        pass

    @abstractmethod
    def on_validation_epoch_start(self):
        pass

    def _common_epoch_end(self, stage: str) -> None:
        batch_metrics = self.steps_metrics_storage.inverse_nest()[stage]
        mean_metrics = {name: sum(values) / len(values) for name, values in batch_metrics.items()}
        msg = [f"Epoch: {self.current_epoch}"]
        for name, value in mean_metrics.items():
            msg.append(f"{stage}/{name}: {round(value, 3)}")
        log.info("  ".join(msg))
        self.epochs_metrics_storage.append(mean_metrics, split=stage)

    def on_train_epoch_end(self) -> None:
        self._common_epoch_end("train")
        for name, scheduler in self.schedulers.items():
            scheduler.step()

    def on_validation_epoch_end(self) -> None:
        self._common_epoch_end("val")

    def on_epoch_end(self):
        optizers_lr = {
            f"{name}_LR": optimizer.param_groups[0]["lr"]
            for name, optimizer in self.optimizers.items()
        }
        self.epochs_metrics_storage.append(optizers_lr, split="train")
