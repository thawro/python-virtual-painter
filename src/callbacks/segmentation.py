"""Training callbacks."""

from src.logging import get_pylogger
from src.visualization import plot_segmentation_results
from src.callbacks.base import BaseCallback
from src.model.module.trainer import Trainer

from typing import Callable

log = get_pylogger(__name__)


class SegmentationExamplesPlotterCallback(BaseCallback):
    """Plot prediction examples"""

    def __init__(
        self,
        labels: list[str],
        inverse_preprocessing: Callable,
        cmap: list[tuple[int, int, int]],
        stage: str,
        filepath: str | None = None,
    ):
        if filepath is None:
            filepath = "examples.jpg"
        self.filepath = filepath
        self.stage = stage
        self.labels = labels
        self.inverse_preprocessing = inverse_preprocessing
        self.cmap = cmap

    def on_epoch_end(self, trainer: Trainer) -> None:
        results = trainer.module.results[self.stage]
        plot_segmentation_results(results, self.cmap, self.inverse_preprocessing, self.filepath)
