import random
from typing import Tuple, Literal, List, Any

import cv2 as cv
import numpy as np
import numpy.typing as npt

Frame_t = npt.NDArray[np.uint8]
Pixel_t = npt.NDArray[np.uint8]


class VideoEditor:
    def __init__(self) -> None:
        self.video_list: List[str] = []
        self.effect_queue: List[Tuple[Any, ...]] = []
        self.cut_parts: List[Tuple[float, float]] = []
        self.errors: List[str] = []

    def add_video(self, path: str) -> 'VideoEditor':
        self.video_list.append(path)
        return self

    def grayscale(self, start: float, end: float) -> 'VideoEditor':
        self.effect_queue.append(("g", start, end))
        return self

    def chromakey(self, start: float, end: float, img: str, color: Tuple[int, int, int],
                  similarity: int) -> 'VideoEditor':
        self.effect_queue.append(("ch", start, end, img, color, similarity))
        return self

    def cut(self, start: float, end: float) -> 'VideoEditor':
        self.cut_parts.append((start, end))
        return self

    def shaky_cam(self, start: float, end: float) -> 'VideoEditor':
        self.effect_queue.append(("s", start, end))
        return self

    def image(self, start: float, end: float, img: str, pos: Tuple[float, float, float, float]) -> 'VideoEditor':
        self.effect_queue.append(("i", start, end, img, pos))
        return self

    def zoom(self, start: float, end: float, pos: Tuple[float, float, float, float]) -> 'VideoEditor':
        self.effect_queue.append(("z", start, end, pos))
        return self

    def flip(self, start: float, end: float, axis: Literal[0, 1, -1]) -> 'VideoEditor':
        self.effect_queue.append(("f", start, end, axis))
        return self

    def rotate(self, start: float, end: float, rotation: int) -> 'VideoEditor':
        self.effect_queue.append(("r", start, end, rotation))
        return self

    def blur(self, start: float, end: float, intensity: int) -> 'VideoEditor':
        self.effect_queue.append(("b", start, end, intensity))
        return self

    def glitch(self, start: float, end: float) -> 'VideoEditor':
        self.effect_queue.append(("gl", start, end))
        return self

    def scan_lines(self, start: float, end: float) -> 'VideoEditor':
        self.effect_queue.append(("sl", start, end))
        return self

    def snow(self, start: float, end: float) -> 'VideoEditor':
        self.effect_queue.append(("sn", start, end))
        return self

    def render(self, path: str, width: int, height: int, framerate: float, short: bool = False) -> 'VideoEditor':
        def are_similar(frame_1: Frame_t, frame_2: Frame_t) -> bool:
            if frame_1.shape != frame_2.shape:
                return False
            pixel_differences = np.abs(frame_1 - frame_2)
            total_difference = np.sum(pixel_differences)
            similarity = (765 - total_difference /
                          (frame_1.shape[0] * frame_1.shape[1])) / 765
            return similarity > 0.9

        def apply_effects_to_frame(original_frame: Frame_t, current_frame_number: float,
                                   frame_rate: float) -> Frame_t:
            def apply_grayscale(frame_to_modify):
                modified_frame = cv.cvtColor(
                    frame_to_modify, cv.COLOR_BGR2GRAY)
                return cv.cvtColor(modified_frame, cv.COLOR_GRAY2BGR)

            def apply_chromakey(frame_to_modify: Frame_t, image_path: str, color_rgb: Tuple[int, int, int],
                                similarity: int) -> Frame_t:
                image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
                if image is None:
                    image_message = "Image at: " + image_path + " not found."
                    if image_message not in self.errors:
                        self.errors.append(image_message)
                    return frame_to_modify
                else:
                    resized_image = cv.resize(
                        image, (frame_to_modify.shape[1], frame_to_modify.shape[0]))
                    color_bgr = np.flip(color_rgb)
                    modified_frame = np.copy(frame_to_modify)
                    pixel_differences = np.sum(
                        np.abs(frame_to_modify - color_bgr), axis=2)
                    mask = pixel_differences < similarity
                    modified_frame[mask] = resized_image[mask]
                    return modified_frame

            def apply_shaky_cam(frame_to_modify: Frame_t) -> Frame_t:
                modified_frame = np.copy(frame_to_modify)
                shift_x = random.choice((-10, 10))
                shift_y = random.choice((-10, 10))
                modified_frame = np.roll(modified_frame, shift_x, axis=0)
                modified_frame = np.roll(modified_frame, shift_y, axis=1)
                return cv.resize(modified_frame, (frame_to_modify.shape[1], frame_to_modify.shape[0]))

            def apply_image(frame_to_modify: Frame_t, image_path: str,
                            position: Tuple[float, float, float, float]) -> Frame_t:
                image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
                if image is None:
                    image_message = "Image at: " + image_path + " not found."
                    if image_message not in self.errors:
                        self.errors.append(image_message)
                    return frame_to_modify
                else:
                    image_width = round(
                        frame_to_modify.shape[1] * (position[2] - position[0]))
                    image_height = round(
                        frame_to_modify.shape[0] * (position[3] - position[1]))
                    resized_image = cv.resize(
                        image, (image_width, image_height))
                    modified_frame = np.copy(frame_to_modify)
                    x1 = round(position[1] * modified_frame.shape[0])
                    x2 = round(position[3] * modified_frame.shape[0])
                    y1 = round(position[0] * modified_frame.shape[1])
                    y2 = round(position[2] * modified_frame.shape[1])
                    if image.shape[2] == 4:
                        mask = resized_image[:, :, 3] > 0
                        modified_frame[x1: x2,
                                       y1: y2][mask] = resized_image[:, :, :3][mask]
                    else:
                        modified_frame[x1: x2, y1: y2] = resized_image
                    return modified_frame

            def apply_zoom(frame_to_modify: Frame_t, position: Tuple[float, float, float, float]) -> Frame_t:
                modified_frame = np.copy(frame_to_modify)
                x1 = round(position[1] * modified_frame.shape[0])
                x2 = round(position[3] * modified_frame.shape[0])
                y1 = round(position[0] * modified_frame.shape[1])
                y2 = round(position[2] * modified_frame.shape[1])
                modified_frame = modified_frame[x1: x2, y1: y2]
                return cv.resize(modified_frame, (frame_to_modify.shape[1], frame_to_modify.shape[0]))

            def apply_flip(frame_to_modify: Frame_t, axis: Literal[0, 1, -1]) -> Frame_t:
                return cv.flip(frame_to_modify, axis)

            def apply_rotate(frame_to_modify: Frame_t, rotation: int) -> Frame_t:
                # Tu som sa trochu inšpiroval chatGPT
                center = (frame_to_modify.shape[1] / 2,
                          frame_to_modify.shape[0] / 2)
                matrix = cv.getRotationMatrix2D(center, rotation, 1)
                modified_frame = cv.warpAffine(frame_to_modify, matrix, (frame_to_modify.shape[1],
                                                                         frame_to_modify.shape[0]))
                # Odtiaľto je zase čisto môj kód
                mask = (np.sum(frame_to_modify, axis=2) != 0) & (
                    np.sum(modified_frame, axis=2) == 0)
                snow = cv.cvtColor(np.random.randint(0, 256, frame_to_modify.shape[:2], dtype=np.uint8),
                                   cv.COLOR_GRAY2BGR)
                modified_frame[mask] = snow[mask]
                return modified_frame

            def apply_blur(frame_to_modify: Frame_t, intensity: int) -> Frame_t:
                kernel_size = intensity + 2
                kernel = np.ones((kernel_size, kernel_size),
                                 np.uint8) / (kernel_size ** 2)
                modified_frame = cv.filter2D(
                    np.copy(frame_to_modify), -1, kernel)
                return modified_frame

            def apply_glitch(frame_to_modify: Frame_t) -> Frame_t:
                modified_frame = np.copy(frame_to_modify)
                shift = random.randint(-2, 2)
                modified_frame = np.roll(modified_frame, shift, axis=2)
                block_size_x = frame_to_modify.shape[1] // 10
                block_size_y = frame_to_modify.shape[0] // 10
                for i in range(0, frame_to_modify.shape[0], block_size_y):
                    for j in range(0, frame_to_modify.shape[1], block_size_x):
                        block = modified_frame[i:i +
                                               block_size_y, j:j + block_size_x]
                        np.random.shuffle(block)
                        modified_frame[i:i + block_size_y,
                                       j:j + block_size_x] = block
                return modified_frame

            def apply_scan_lines(frame_to_modify: Frame_t, number: float) -> Frame_t:
                modified_frame = np.copy(frame_to_modify)
                shift = round(number) % 2
                for i in range(shift, frame_to_modify.shape[0]):
                    scan_line = np.random.randint(
                        0, 256, (frame_to_modify.shape[1], 3), dtype=np.uint8)
                    transparency = random.randint(3, 9) / 10
                    modified_frame[i] = cv.addWeighted(
                        modified_frame[i], transparency, scan_line, 1 - transparency, 0)
                return modified_frame

            def apply_snow(frame_to_modify: Frame_t) -> Frame_t:
                modified_frame = np.random.randint(
                    0, 256, frame_to_modify.shape[:2], dtype=np.uint8)
                return cv.cvtColor(modified_frame, cv.COLOR_GRAY2BGR)

            current_second = current_frame_number / frame_rate
            frame_to_process = np.copy(original_frame)
            for effect in self.effect_queue:
                if effect[1] <= current_second < effect[2]:
                    if effect[0] == "g":
                        frame_to_process = apply_grayscale(frame_to_process)
                    elif effect[0] == "ch":
                        frame_to_process = apply_chromakey(
                            frame_to_process, effect[3], effect[4], effect[5])
                    elif effect[0] == "s":
                        frame_to_process = apply_shaky_cam(frame_to_process)
                    elif effect[0] == "i":
                        frame_to_process = apply_image(
                            frame_to_process, effect[3], effect[4])
                    elif effect[0] == "z":
                        frame_to_process = apply_zoom(
                            frame_to_process, effect[3])
                    elif effect[0] == "f":
                        frame_to_process = apply_flip(
                            frame_to_process, effect[3])
                    elif effect[0] == "r":
                        frame_to_process = apply_rotate(
                            frame_to_process, effect[3])
                    elif effect[0] == "b":
                        frame_to_process = apply_blur(
                            frame_to_process, effect[3])
                    elif effect[0] == "gl":
                        frame_to_process = apply_glitch(frame_to_process)
                    elif effect[0] == "sl":
                        frame_to_process = apply_scan_lines(
                            frame_to_process, current_frame_number)
                    elif effect[0] == "sn":
                        frame_to_process = apply_snow(frame_to_process)
            return frame_to_process

        final_video = cv.VideoWriter(path, cv.VideoWriter.fourcc(
            *"mp4v"), framerate, (width, height), True)
        frame_number = 0.0
        previous_frame = np.zeros(1, dtype=np.uint8)

        for video in self.video_list:
            current_video = cv.VideoCapture(video)
            current_frame_rate = current_video.get(cv.CAP_PROP_FPS)
            if current_frame_rate == 0.0:
                message = "Video at: " + video + " not found."
                if message not in self.errors:
                    self.errors.append(message)
            else:
                frame_rate_ratio = framerate / current_frame_rate
                frames_to_process = 0.0
                while True:
                    cut = False
                    for part in self.cut_parts:
                        if part[0] <= frame_number / framerate < part[1]:
                            cut = True
                    ret, frame = current_video.read()
                    if not ret:
                        break
                    processed_frame = apply_effects_to_frame(
                        cv.resize(frame, (width, height)), frame_number, framerate)
                    if not (short and are_similar(processed_frame, previous_frame)) and not cut:
                        previous_frame = processed_frame
                        frames_to_process += frame_rate_ratio
                        while frames_to_process >= 1:
                            final_video.write(processed_frame)
                            cv.imshow("frame", processed_frame)
                            if cv.waitKey(1) == ord('q'):
                                break
                            frames_to_process -= 1
                            frame_number += 1
                            if short:
                                frames_to_process %= 1
                    elif cut:
                        frame_number += frame_rate_ratio
                current_video.release()
        final_video.release()
        cv.destroyAllWindows()
        for error in self.errors:
            print(error)
        return self


if __name__ == "__main__":
    VideoEditor().add_video("C:/Users/samko/Documents/Python/KSI23/2.Vlna/velke_ulohy/VideoEditor/ukazky/clean.mp4").cut(5, 10).cut(17.5, 20).render("video_cut.mp4", 426, 240, 25)
