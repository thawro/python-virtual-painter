"""Transforms for segmentation task"""

import albumentations as A
import numpy as np
import cv2
from abc import abstractmethod
from albumentations.pytorch import ToTensorV2
from typing import Callable

_mean_std = float | tuple[float, float, float]


class ImageTransform:
    def __init__(
        self,
        imgsz: int = 256,
        mean: _mean_std = (0.485, 0.456, 0.406),
        std: _mean_std = (0.229, 0.224, 0.225),
    ):
        self.imgsz = imgsz
        self.mean = mean
        self.std = std

    @property
    @abstractmethod
    def train(self) -> Callable:
        raise NotImplementedError()

    @property
    @abstractmethod
    def inference(self) -> Callable:
        raise NotImplementedError()

    @property
    def inverse_preprocessing(self):
        """Apply inverse of preprocessing to the image (for visualization purposes)."""

        def transform(image: np.ndarray) -> np.ndarray:
            _image = (image * np.array(self.std)) + np.array(self.mean)
            return (_image * 255).astype(np.uint8)

        return transform


class SegmentationTransform(ImageTransform):
    orig_masks_size = 512

    @property
    def train(self):
        return A.Compose(
            [
                A.Resize(self.orig_masks_size, self.orig_masks_size),
                A.RandomScale((-0.5, 1.0), p=0.8),
                A.Rotate((-10, 10), p=0.8),
                A.GaussianBlur(p=0.5),
                A.PadIfNeeded(
                    min_height=self.imgsz, min_width=self.imgsz, border_mode=cv2.BORDER_CONSTANT
                ),
                A.RandomCrop(height=self.imgsz, width=self.imgsz),
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
                ToTensorV2(),
            ],
            is_check_shapes=False,
        )

    @property
    def inference(self):
        return A.Compose(
            [
                A.Resize(self.imgsz, self.imgsz),
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=255),
                ToTensorV2(),
            ],
            is_check_shapes=False,
        )
