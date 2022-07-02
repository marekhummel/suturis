import cv2
import numpy as np


def highlight_features(image, keypoints, matches, status, kpsIndex):
    copy = np.copy(image)
    red = (0, 0, 255)
    green = (0, 255, 0)
    # loop over the matches
    for (indices, s) in zip(matches, status):
        keypoint = keypoints[indices[kpsIndex]]
        center = (int(keypoint[0]), int(keypoint[1]))
        cv2.circle(copy, center, 6, green if s else red, 1)
    return copy


def draw_matches(imageA, imageB, kpsA, kpsB, matches, status):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((hA + hB, max(wA, wB), 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[hA:, 0:wB] = imageB
    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]), int(kpsB[trainIdx][1]) + hA)
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    # return the visualization
    return vis


def concat_images(image1, image2, axis=0):
    image1 = cv2.resize(image1, (0, 0), None, 0.5, 0.5)
    image2 = cv2.resize(image2, (0, 0), None, 0.5, 0.5)
    return np.concatenate((image1, image2), axis=0)
