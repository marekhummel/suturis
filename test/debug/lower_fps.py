import cv2

TARGET_FPS = 2
PATH = "data/examples/files/2022-05-28/"
FILENAME = "bow_0120220528105828.mp4"

video = cv2.VideoCapture(PATH + FILENAME)
src_fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
target = PATH + FILENAME.replace(".mp4", "_modified.mp4")
writer = cv2.VideoWriter(target, fourcc, TARGET_FPS, (width, height))

ratio = int(src_fps // TARGET_FPS)
print(ratio)
while True:
    success, frame = video.read()
    if not success:
        break

    writer.write(frame)

    for _ in range(ratio - 1):
        success, frame = video.read()
