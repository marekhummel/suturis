from black import detect_target_versions
import cv2
import numpy as np


def detect_lines_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 150
    high_threshold = 250
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    rho = 1               # distance resolution in pixels of the Hough grid
    theta = np.pi / 180     # angular resolution in radians of the Hough grid
    threshold = 15          # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 180    # minimum number of pixels making up a line
    max_line_gap = 30       # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(
        edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap
    )

    # print(len(lines))
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # Draw the lines on the  image
    return image
    # return line_image
    # return cv2.addWeighted(image, 0.8, line_image, 1, 0)


def detect_lines_lsd(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    line_image = np.copy(image)
    detector = cv2.createLineSegmentDetector()
    lines = detector.detect(gray)[0]
    detector.drawSegments(line_image, lines)

    return line_image
