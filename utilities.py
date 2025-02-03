import numpy as np
from cv2.typing import MatLike


def are_frames_similar(frame_1: MatLike, frame_2: MatLike | None) -> bool:
    if frame_2 is None or frame_1.shape != frame_2.shape:
        return False
    pixel_differences = np.abs(frame_1 - frame_2)
    total_difference = np.sum(pixel_differences)
    similarity: float = (765 - total_difference /
                         (frame_1.shape[0] * frame_1.shape[1])) / 765
    return similarity > 0.9
