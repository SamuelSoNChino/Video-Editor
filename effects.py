import cv2 as cv
import numpy as np
import numpy.typing as npt

Frame_t = npt.NDArray[np.uint8]
Pixel_t = npt.NDArray[np.uint8]
Color = tuple[int, int, int]


class Efect():
    pass


class GrayscaleEffect(Efect):
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def apply(self, frame: Frame_t, current_second: float) -> Frame_t:
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

    def apply(self, frame: Frame_t, current_second: float) -> Frame_t:
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
