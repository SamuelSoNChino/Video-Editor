import cv2 as cv
from utilities import are_frames_similar
from effects import Position, \
    Color, \
    Effect, \
    EffectRenderer, \
    GrayscaleEffect, \
    ChromakeyEffect, \
    ShakyCamEffect, \
    ImageEffect, \
    ZoomEffect, \
    FlipEffect, \
    RotationEffect, \
    BlurEffect, \
    GlitchEffect, \
    ScanLinesEffect, \
    SnowEffect


class VideoEditor:
    def __init__(self) -> None:
        self.video_list: list[str] = []
        self.effect_queue: list[Effect] = []
        self.cut_parts: list[tuple[float, float]] = []

    def add_video(self, path: str) -> 'VideoEditor':
        self.video_list.append(path)
        return self

    def cut(self, start: float, end: float) -> 'VideoEditor':
        self.cut_parts.append((start, end))
        return self

    def render(self, path: str, width: int, height: int,
               frame_rate: float, short: bool = False,
               show_preview: bool = True) -> 'VideoEditor':

        final_video = cv.VideoWriter(path, cv.VideoWriter.fourcc(
            *"mp4v"), frame_rate, (width, height), True)
        effect_renderer = EffectRenderer(self.effect_queue)
        frame_number = 0.0
        previous_frame = None

        try:
            for video in self.video_list:
                current_video = cv.VideoCapture(video)
                current_frame_rate = current_video.get(cv.CAP_PROP_FPS)
                if not current_video.isOpened():
                    raise ValueError(f"Error opening video file: {video}")
                frame_rate_ratio = frame_rate / current_frame_rate
                frames_to_process = 0.0
                while True:
                    ret, frame = current_video.read()
                    if not ret:
                        break

                    current_second = frame_number / frame_rate

                    is_cut = False
                    for start, end in self.cut_parts:
                        if start <= current_second < end:
                            is_cut = True
                            break
                    if is_cut:
                        frame_number += frame_rate_ratio
                        continue

                    frame = cv.resize(frame, (width, height))
                    frame = effect_renderer.render_frame(frame, current_second)
                    if not short or not are_frames_similar(frame,
                                                           previous_frame):
                        previous_frame = frame
                        frames_to_process += frame_rate_ratio
                        while frames_to_process >= 1:
                            final_video.write(frame)
                            if show_preview and frame_number % 10 == 0:
                                cv.imshow("frame", frame)
                                cv.waitKey(round(1000 / frame_rate * 10))
                            frames_to_process -= 1
                            frame_number += 1
                            if short:
                                frames_to_process %= 1

                current_video.release()
        finally:
            final_video.release()
            cv.destroyAllWindows()
            return self

    def grayscale(self, start: float, end: float) -> 'VideoEditor':
        self.effect_queue.append(GrayscaleEffect(start, end))
        return self

    def chromakey(self, start: float, end: float, image_path: str,
                  color: Color, similarity: int) -> 'VideoEditor':
        self.effect_queue.append(ChromakeyEffect(
            start, end, image_path, color, similarity))
        return self

    def shaky_cam(self, start: float, end: float) -> 'VideoEditor':
        self.effect_queue.append(ShakyCamEffect(start, end))
        return self

    def image(self, start: float, end: float, img: str,
              pos: Position) -> 'VideoEditor':
        self.effect_queue.append(ImageEffect(start, end, img, pos))
        return self

    def zoom(self, start: float, end: float,
             pos: Position) -> 'VideoEditor':
        self.effect_queue.append(ZoomEffect(start, end, pos))
        return self

    def flip(self, start: float, end: float,
             axis: int) -> 'VideoEditor':
        self.effect_queue.append(FlipEffect(start, end, axis))
        return self

    def rotate(self, start: float, end: float, rotation: int) -> 'VideoEditor':
        self.effect_queue.append(RotationEffect(start, end, rotation))
        return self

    def blur(self, start: float, end: float, intensity: int) -> 'VideoEditor':
        self.effect_queue.append(BlurEffect(start, end, intensity))
        return self

    def glitch(self, start: float, end: float) -> 'VideoEditor':
        self.effect_queue.append(GlitchEffect(start, end))
        return self

    def scan_lines(self, start: float, end: float) -> 'VideoEditor':
        self.effect_queue.append(ScanLinesEffect(start, end))
        return self

    def snow(self, start: float, end: float) -> 'VideoEditor':
        self.effect_queue.append(SnowEffect(start, end))
        return self


if __name__ == "__main__":
    pass
