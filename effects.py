import random
import cv2 as cv
from cv2.typing import MatLike
import numpy as np


Color = tuple[int, int, int]
Position = tuple[float, float, float, float]


class Effect():
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def applies_to(self, current_second: float) -> bool:
        return self.applies_to(current_second)

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        raise NotImplementedError(
            "Effect subclasses must implement 'apply' method")


class EffectRenderer():
    def __init__(self, effect_queue: list[Effect]):
        self.effect_queue = effect_queue

    def render_frame(self, frame: MatLike, current_second: float) -> MatLike:
        for effect in self.effect_queue:
            frame = effect.apply(frame, current_second)
        return frame


class GrayscaleEffect(Effect):
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.applies_to(current_second):
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        return frame


class ChromakeyEffect(Effect):
    def __init__(self, start: float, end: float, image_path: str,
                 color: Color, similarity: int):
        self.start = start
        self.end = end
        self.image_path = image_path
        self.color = np.flip(color)  # Convert RGB to BGR
        self.similarity = similarity
        self.image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        if self.image is None:
            raise FileNotFoundError(
                f'ERROR: Image at: {image_path} not found.')

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.applies_to(current_second):
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


class ShakyCamEffect(Effect):
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.applies_to(current_second):
            shift_x = random.choice((-10, 10))
            shift_y = random.choice((-10, 10))
            frame = np.roll(frame, shift_x, axis=0)
            frame = np.roll(frame, shift_y, axis=1)
            frame = cv.resize(frame, (frame.shape[1], frame.shape[0]))
        return frame


class ImageEffect(Effect):
    def __init__(self, start: float, end: float,
                 image_path: str, position: Position):
        self.start = start
        self.end = end
        self.image_path = image_path
        self.position = position  # (x_min, y_min, x_max, y_max)
        self.image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        if self.image is None:
            raise FileNotFoundError(
                f'ERROR: Image at: {image_path} not found.')

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.applies_to(current_second):
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


class ZoomEffect(Effect):
    def __init__(self, start: float, end: float, position: Position):
        self.start = start
        self.end = end
        self.position = position  # (x_min, y_min, x_max, y_max)

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.applies_to(current_second):

            frame_height, frame_width = frame.shape[:2]
            x_min = round(self.position[0] * frame_width)
            y_min = round(self.position[1] * frame_height)
            x_max = round(self.position[2] * frame_width)
            y_max = round(self.position[3] * frame_height)

            zoomed_frame = frame[y_min: y_max, x_min: x_max]
            frame = cv.resize(zoomed_frame,
                              (frame.shape[1], frame.shape[0]))

        return frame


class FlipEffect(Effect):
    def __init__(self, start: float, end: float, axis: int):
        self.start = start
        self.end = end
        self.axis = axis

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.applies_to(current_second):
            frame = cv.flip(frame, self.axis)
        return frame


class RotationEffect(Effect):
    def __init__(self, start: float, end: float, rotation: int):
        self.start = start
        self.end = end
        self.rotation = rotation

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.applies_to(current_second):
            center = (frame.shape[1] / 2, frame.shape[0] / 2)
            matrix = cv.getRotationMatrix2D(center, self.rotation, 1)
            rotated_frame = cv.warpAffine(frame, matrix,
                                          (frame.shape[1], frame.shape[0]))
            gray_snow = np.random.randint(0, 256, frame.shape[:2],
                                          dtype=np.uint8)
            snow = cv.cvtColor(gray_snow, cv.COLOR_GRAY2BGR)
            mask = (np.sum(frame, axis=2) != 0) & (np.sum(frame, axis=2) == 0)
            rotated_frame[mask] = snow[mask]
            frame = rotated_frame
        return frame


class BlurEffect(Effect):
    def __init__(self, start: float, end: float, intensity: int):
        self.start = start
        self.end = end
        self.intensity = intensity

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.applies_to(current_second):
            kernel_size = self.intensity + 2
            kernel = np.ones((kernel_size, kernel_size),
                             np.uint8) / (kernel_size ** 2)
            frame = cv.filter2D(frame, -1, kernel)
        return frame


class GlitchEffect(Effect):
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.applies_to(current_second):
            shift = random.randint(-2, 2)
            rolled_frame = np.roll(frame, shift, axis=2)
            block_size_x = frame.shape[1] // 10
            block_size_y = frame.shape[0] // 10
            for i in range(0, frame.shape[0], block_size_y):
                for j in range(0, frame.shape[1], block_size_x):
                    block = rolled_frame[i:i + block_size_y,
                                         j:j + block_size_x]
                    np.random.shuffle(block)
                    frame[i:i + block_size_y,
                          j:j + block_size_x] = block
        return frame


class ScanLinesEffect(Effect):
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.applies_to(current_second):
            shift = random.randint(0, 3)
            for i in range(shift, frame.shape[0]):
                scan_line = np.random.randint(0, 256,
                                              (frame.shape[1], 3),
                                              dtype=np.uint8)
                transparency = random.randint(3, 9) / 10
                frame[i] = cv.addWeighted(frame[i], transparency,
                                          scan_line,
                                          1 - transparency, 0)
        return frame


class SnowEffect(Effect):
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def apply(self, frame: MatLike, current_second: float) -> MatLike:
        if self.applies_to(current_second):
            gray_snow = np.random.randint(0, 256, frame.shape[:2],
                                          dtype=np.uint8)
            frame = cv.cvtColor(gray_snow, cv.COLOR_GRAY2BGR)
        return frame
