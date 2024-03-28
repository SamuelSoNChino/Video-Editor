# Video-Editor
The program allows you to edit videos on a basic level using methods of a VideoEditor class.

## HOW TO USE

The methods are to be used in a chain, with first method being add_video and last one being render. An example use: VideoEditor().add_video("lol.mp4").add_video("video2.mp4").shaky_cam(2,5).render("combination.mp4", 360, 178, 30, short=False)
The effects will be executed sequentially.

### INDIVIDUAL METHODS

add_video(path: str)

cut(start: float, end: float) (in seconds)

grayscale(start: float, end: float)

image(start: float, end: float, img: str, pos: Tuple[float, float, float, float]) (pos: width_start, height_start, width_stop, height_stop; in %)

chromakey(start: float, end: float, img: str, color: Tuple[int, int, int], similarity: int) (color is the one we are trying to filter out)

shaky_cam(start: float, end: float) 

zoom(start: float, end: float, pos: Tuple[float, float, float, float]) (works same as image)

flip(start: float, end: float, axis: Literal[0, 1, -1]) (0 - vertically, 1 - horizontally, -1 - both)

rotate(start: float, end: float, rotation: int) (in degrees)

blur(start: float, end: float, intensity: int) (recommended range for intensity is 1 - 50)

glitch(start: float, end: float)

scan_lines(start: float, end: float)

snow(start: float, end: float)

render(path: str, width: int, height: int, framerate: float, short: bool) (Short makes the whole video shorter by deleting similar frames in a row)


## PROGRAM EXPLAINED

The program is based on builder pattern, with methods always returning the class itself, to allow chain editing. The editing process works sequentially, frame by frame, applying efects or duplicating or removing frames when needed. The effects are mostly implemented with simple calculations using numpy and opencv.
