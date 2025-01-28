# Video-Editor

A lightweight Python-based video editing tool using OpenCV and NumPy. This program provides a simple interface for applying basic video effects, trimming, and exporting videos. Designed with a builder pattern, the VideoEditor class allows for seamless method chaining, making video editing straightforward and intuitive.

[Quick Showcase of the project being used with Terminal](https://youtu.be/qym4_PWdxm8)

>This was one of my earlier projects, so there are parts I would refactor. For example, the monolithic `render` method. I might look into it in the future.

## Features

- Chainable methods for clean, expressive editing pipelines.

- Support for multiple effects, including:

    - Grayscale, chromakey, blur, zoom, and rotate.

    - Fun effects like shaky cam, glitch, scan lines, and snow.

- Trim videos by cutting specific sections.

- Add images or overlays to videos.

- Optimize video size by removing similar consecutive frames.

- Compatible with MP4 output format.

## Installation

**Clone the repository**:

    git clone https://github.com/SamuelSoNChino/Video-Editor.git
    cd Video-Editor

**Install dependencies**:

    pip install numpy opencv-python

## Quick Start

    from video_editor import VideoEditor

    # Example usage:
    VideoEditor() \
        .add_video("input.mp4") \
        .grayscale(0, 10) \
        .shaky_cam(5, 7) \
        .cut(15, 20) \
        .render("output.mp4", 640, 360, 30)

### Chainable Editing Methods

- `add_video(path: str)`

    Adds a video to the editing pipeline.

- `cut(start: float, end: float)`

    Trims a specific section of the video (in seconds).

- `grayscale(start: float, end: float)`

    Converts frames to grayscale within the specified time range.

- `chromakey(start: float, end: float, img: str, color: Tuple[int, int, int], similarity: int)`

    Replaces a specific color in the video with a given image.

    Parameters:

    `img`: Path to the replacement image.

    `color`: RGB tuple of the color to filter.

    `similarity`: Threshold for color matching.


- `shaky_cam(start: float, end: float)`
    
    Adds a "shaky camera" effect to simulate motion.

- `zoom(start: float, end: float, pos: Tuple[float, float, float, float])`

    Zooms into a specific area of the frame.

    Parameters:

    `pos`: Tuple representing (x_start, y_start, x_end, y_end) as percentages.

- `image(start: float, end: float, img: str, pos: Tuple[float, float, float, float])`


    Overlays an image on the video at a specific position.

    Parameters:

    `pos`: Tuple representing (x_start, y_start, x_end, y_end) as percentages.

- `flip(start: float, end: float, axis: Literal[0, 1, -1])`

    Flips the video:

    `0`: Vertically

    `1`: Horizontally

    `-1`: Both

- `rotate(start: float, end: float, rotation: int)`

    Rotates the video by a specified angle (in degrees).

- `blur(start: float, end: float, intensity: int)`

    Blurs frames within the given time range.

    Recommended intensity: 1–50.

- `glitch(start: float, end: float)`
    
    Adds a random glitch effect.

- `scan_lines(start: float, end: float)`

    Adds scan lines to create a retro effect.

- `snow(start: float, end: float)`

    Simulates snow-like static noise.

- `render(path: str, width: int, height: int, framerate: float, short: bool = False)`

    Exports the final video to the specified path.

    Parameters:

    `short`: Removes similar frames to reduce file size.

## Example Pipelines

Using the builder pattern:

    VideoEditor() \
        .add_video("video1.mp4") \
        .add_video("video2.mp4") \
        .grayscale(0, 5) \
        .chromakey(5, 10, "background.png", (0, 255, 0), 50) \
        .shaky_cam(10, 15) \
        .blur(15, 20, intensity=10) \
        .render("final_output.mp4", width=1280, height=720, framerate=30)

Or using methods individually in Python interpreter:

    >>> ve = VideoEditor()
    >>> ve.add_video("video1.mp4")
    >>> ve.add_video("video2.mp4")
    >>> ve.grayscale(0, 5)
    >>> ve.chromakey(5, 10, "background.png", (0, 255, 0), 50)
    >>> ve.shaky_cam(10, 15)
    >>> ve.blur(15, 20, intensity=10)
    >>> ve.render("final_output.mp4", width=1280, height=720, framerate=30)


## Implementation Details

- Built using OpenCV for video processing and NumPy for efficient matrix operations.

- Effects are applied frame-by-frame to ensure precision and flexibility.

- Supports real-time preview of the rendering process (press q to exit preview).
