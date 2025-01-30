import random
import cv2 as cv
from cv2.typing import MatLike
import numpy as np


Color = tuple[int, int, int]
Position = tuple[float, float, float, float]


class Efect():
    pass


class GrayscaleEffect(Efect):
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.start <= current_second < self.end:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        return frame


class ChromakeyEffect(Efect):
    def __init__(self, start: float, end: float, image_path: str,
                 color: Color, similarity: int):
        self.start = start
        self.end = end
        self.image_path = image_path
        self.color = np.flip(color)  # Convert RGB to BGR
        self.similarity = similarity
        self.image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        if self.image is None:
            print(f'ERROR: Image at: {image_path} not found.')

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.start <= current_second < self.end:
            if self.image is None:
                return frame  # Skip if image is not loaded
            resized_image = cv.resize(self.image,
                                      (frame.shape[1],
                                       frame.shape[0]))
            pixel_differences = np.sum(
                np.abs(frame - self.color), axis=2)
            mask = pixel_differences < self.similarity
            frame[mask] = resized_image[mask]
        return frame


class ShakyCamEffect(Efect):
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.start <= current_second < self.end:
            shift_x = random.choice((-10, 10))
            shift_y = random.choice((-10, 10))
            frame = np.roll(frame, shift_x, axis=0)
            frame = np.roll(frame, shift_y, axis=1)
            frame = cv.resize(frame, (frame.shape[1], frame.shape[0]))
        return frame


class ImageEffect(Efect):
    def __init__(self, start: float, end: float,
                 image_path: str, position: Position):
        self.start = start
        self.end = end
        self.image_path = image_path
        self.position = position  # (x_min, y_min, x_max, y_max)
        self.image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        if self.image is None:
            print(f'ERROR: Image at: {image_path} not found.')

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.start <= current_second < self.end:
            if self.image is None:
                return frame

            frame_height, frame_width = frame.shape[:2]
            x_min = round(self.position[0] * frame_width)
            y_min = round(self.position[1] * frame_height)
            x_max = round(self.position[2] * frame_width)
            y_max = round(self.position[3] * frame_height)

            target_width = x_max - x_min
            target_height = y_max - y_min
            resized_image = cv.resize(self.image,
                                      (target_width, target_height))

            if resized_image.shape[2] == 4:  # Check for alpha channel
                mask = resized_image[:, :, 3] > 0
                frame[y_min:y_max, x_min:x_max][mask] = \
                    resized_image[:, :, :3][mask]
            else:
                frame[y_min:y_max, x_min:x_max] = resized_image
        return frame
