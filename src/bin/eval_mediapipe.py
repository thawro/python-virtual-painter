"""Evaluate the model"""
import cv2
import numpy as np
from PIL import Image
import math
from src.utils.video import process_video, save_frames_to_video, get_video_size
from src.utils.image import GREEN
import time

from mediapipe.python.solutions.hands import Hands, HandLandmark


def parse_landmark(landmark, h, w):
    x = int(w * landmark.x)
    y = int(h * landmark.y)
    z = landmark.z
    return x, y, z


def calc_dist(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return math.hypot(abs(x2 - x1), abs(y2 - y1))


def center_point(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    xc = int((x1 + x2) / 2)
    yc = int((y1 + y2) / 2)
    return xc, yc


def is_in_range(pt, x1, y1, x2, y2):
    x_pt, y_pt = pt
    return x_pt <= x2 and x_pt >= x1 and y_pt <= y2 and y_pt >= y1


MODE = "COLOR SET"
BRUSH_SIZE = 10
MIN_DIST = 40
GREY = (110, 110, 110)
H, W = 480, 640
FRAMES = []
is_brush_set = False

palette = np.array(Image.open("palette.jpeg"))
palette = cv2.rotate(palette, cv2.ROTATE_90_CLOCKWISE)

saturation = np.array(Image.open("saturation.png"))
saturation = cv2.resize(saturation, (225, 225))

eraser = np.array(Image.open("eraser.png"))
mask = eraser[..., -1] != 0
eraser = eraser[..., :3]
eraser[~mask] = np.array(GREY)
eraser = cv2.resize(eraser, (225, 225))

# make gradient
palette = (palette * saturation.mean(2, keepdims=True) / 255).astype(np.uint8)


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 1)]
CURRENT_COLOR = colors[-1]
color_ranges = []

ncolors = len(colors) + 1
marg_lr = 30
marg_color = 20
color_width = (W - 2 * marg_lr) // ncolors - marg_color
color_height = 60

eraser_x1, eraser_y1 = marg_lr, 10
eraser_x2, eraser_y2 = eraser_x1 + color_width, eraser_y1 + color_height
eraser_range = (eraser_x1, eraser_y1, eraser_x2, eraser_y2)

for i, color in enumerate(colors, start=1):
    x1, y1 = i * (color_width + marg_color) + marg_lr, 10
    x2, y2 = x1 + color_width, y1 + color_height
    color_ranges.append([x1, y1, x2, y2])


blank = np.zeros((480, 640, 3), dtype=np.uint8)
mask = (blank.any(axis=2) * 255).astype(np.uint8)

s_time = time.time()
prev_pt = None
# For webcam input:
cap = cv2.VideoCapture(0)
hands = Hands(
    model_complexity=0, max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75
)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    h, w, c = image.shape
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (w, 80), GREY, -1)
    for color, (x1, y1, x2, y2) in zip(colors, color_ranges):
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    image[eraser_y1:eraser_y2, eraser_x1:eraser_x2] = cv2.resize(
        eraser, (abs(eraser_x2 - eraser_x1), abs(eraser_y2 - eraser_y1))
    )
    if results.multi_hand_landmarks is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            thumb_tip_lm = landmarks[HandLandmark.THUMB_TIP]
            finger_tip_lm = landmarks[HandLandmark.INDEX_FINGER_TIP]
            middle_tip_lm = landmarks[HandLandmark.MIDDLE_FINGER_TIP]

            thumb_x, thumb_y, thumb_z = parse_landmark(thumb_tip_lm, h, w)
            finger_tip_x, finger_tip_y, finger_tip_z = parse_landmark(finger_tip_lm, h, w)
            middle_tip_x, middle_tip_y, middle_tip_z = parse_landmark(middle_tip_lm, h, w)

            thumb_pt = (thumb_x, thumb_y)
            index_tip_pt = (finger_tip_x, finger_tip_y)
            middle_tip_pt = (middle_tip_x, middle_tip_y)

            thumb_middle_dist = calc_dist(thumb_pt, middle_tip_pt)
            thumb_index_dist = calc_dist(thumb_pt, index_tip_pt)

            draw_x, draw_y = index_tip_pt
            if draw_x > 0 and draw_x < w and thumb_middle_dist < MIN_DIST and draw_y > 80:
                MODE = "DRAW"
                if prev_pt is None:
                    prev_pt = (draw_x, draw_y)
                else:
                    cv2.line(blank, prev_pt, (draw_x, draw_y), CURRENT_COLOR, BRUSH_SIZE)
                    prev_pt = (draw_x, draw_y)
            else:
                prev_pt = None
                MODE = "COLOR SET"

            if thumb_index_dist < MIN_DIST and thumb_middle_dist > MIN_DIST:
                is_brush_set = True
                MODE = "SIZE SET"

            if is_brush_set:
                if thumb_middle_dist < MIN_DIST:
                    is_brush_set = False
                BRUSH_SIZE = math.ceil(thumb_index_dist / 200 * 45)

            for i, (color, (x1, y1, x2, y2)) in enumerate(zip(colors, color_ranges)):
                if is_in_range((draw_x, draw_y), x1, y1, x2, y2):
                    if thumb_middle_dist < MIN_DIST:
                        image[y1:y2, x1:x2] = cv2.resize(palette, (abs(x2 - x1), abs(y2 - y1)))
                    CURRENT_COLOR = tuple(image[draw_y, draw_x].tolist())
                    colors[i] = CURRENT_COLOR
            if is_in_range((draw_x, draw_y), *eraser_range):
                CURRENT_COLOR = (0, 0, 0)
            cv2.circle(image, thumb_pt, 3, GREEN, -1)
            cv2.circle(image, index_tip_pt, (BRUSH_SIZE // 2) + 2, CURRENT_COLOR, 2)
            cv2.circle(image, middle_tip_pt, 3, GREEN, -1)

    mask = blank.any(axis=2)
    image[mask] = blank[mask]
    image = cv2.flip(image, 1)
    e_time = time.time()
    fps = 1 / (e_time - s_time)
    s_time = e_time
    txt = f"FPS: {int(fps)}"
    cv2.putText(image, txt, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    txt = f"mode: {MODE}"
    (txt_w, txt_h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(image, txt, (w - txt_w, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    FRAMES.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.imshow("Frame", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()

save_frames_to_video(FRAMES, 30, "draw.mp4")
