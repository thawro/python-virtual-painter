import numpy as np
import cv2
from typing import Literal

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)


def stack_frames_horizontally(frames: list[np.ndarray], vspace: int = 10, hspace: int = 10):
    img_w = sum([img.shape[1] for img in frames]) + (len(frames) + 1) * vspace
    img_h = max([img.shape[0] for img in frames]) + 2 * hspace
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    ymin = hspace
    xmin = vspace

    for frame in frames:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        h, w = frame.shape[:2]

        if h < img_h - 2 * hspace:
            ymin = img_h // 2 - h // 2
        else:
            ymin = hspace
        img[ymin : ymin + h, xmin : xmin + w, :] = frame
        xmin = xmin + w + vspace
    return img


def add_txt_to_image(
    image: np.ndarray,
    labels: list[str],
    vspace: int = 10,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.5,
    thickness=1,
    loc: Literal["tl", "tc", "tr", "bl", "bc", "br"] = "tl",
):
    img_h, img_w = image.shape[:2]
    txt_h = vspace
    txt_w = 0
    for label in labels:
        (width, height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        txt_h += height + vspace
        txt_w = max(txt_w, width)
    txt_w += 2 * vspace

    if loc == "tl":
        y = vspace
        x = vspace
    elif loc == "tr":
        y = vspace
        x = img_w - txt_w - vspace
    elif loc == "bl":
        y = img_h - txt_h - vspace
        x = vspace
    elif loc == "br":
        y = img_h - txt_h - vspace
        x = img_w - txt_w - vspace
    elif loc == "tc":
        y = vspace
        x = img_w // 2 - txt_w // 2 - vspace
    elif loc == "bc":
        y = img_h - txt_h - vspace
        x = img_w // 2 - txt_w // 2 - vspace

    cv2.rectangle(
        image, (x - vspace, y - vspace), (x - vspace + txt_w, y - vspace + txt_h), GRAY, -1
    )
    for label in labels:
        (width, height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.putText(image, label, (x, y + height), font, font_scale, WHITE)
        y += height + vspace
    return image


def add_labels_to_frames(
    frames: list[np.ndarray],
    labels: list[str],
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.5,
    thickness=1,
):
    labeled_frames = []
    for i in range(len(frames)):
        label = labels[i]
        image = frames[i]
        if isinstance(label, str):
            label = [label]

        labeled_image = add_txt_to_image(
            image.copy(), label, vspace=5, font=font, font_scale=font_scale, thickness=thickness
        )
        labeled_frames.append(labeled_image)
    return labeled_frames
