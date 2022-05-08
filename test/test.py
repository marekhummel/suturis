import sys
import cv2
import numpy as np
import random


def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    img1 = cv2.copyMakeBorder(img1, 10, 1, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img2 = cv2.copyMakeBorder(
        img2, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    r, c = img1.shape[:2]
    r1, c1 = img2.shape[:2]

    # Create a blank image with the size of the first image + second image
    # output_img = np.zeros((max([r, r1]), c+c1, 3), dtype='uint8')
    # output_img[:r, :c, :] = np.dstack([img1, img1, img1])
    # output_img[:r1, c:c+c1, :] = np.dstack([img2, img2, img2])

    output_img = np.concatenate((img1, img2), axis=0)

    # Go over all of the matching points and extract them
    count = 0
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt
        color = (255, 255, 255)
        if count == -1:
            color = (0, 0, 255)
        elif count == -1:
            color = (0, 255, 0)

        # Draw circles on the keypoints
        cv2.circle(output_img, (int(x1), int(y1)), 4, color, cv2.FILLED, 1)
        cv2.circle(output_img, (int(x2), int(y2) + r), 4, color, cv2.FILLED, 1)
        # output_img = cv2.putText(output_img, f"{count}", (int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Connect the same keypoints
        cv2.line(output_img, (int(x1), int(y1)), (int(x2), int(y2) + r), color, 2)

        count += 1
    return output_img


def draw_features(img1, keypoints):
    output_img = img1
    for keypoint in keypoints:
        (x1, y1) = keypoint.pt
        cv2.circle(output_img, (int(x1), int(y1)), 4, (0, 0, 255), cv2.FILLED, 1)
    return output_img


def warpImages(img1, img2, H):

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32(
        [[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]
    ).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(
        -1, 1, 2
    )

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array(
        [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]]
    )

    # ----------------------------------------------------------------------
    # image, where transformed image is on top of the other
    output_img = cv2.warpPerspective(
        img2, H_translation.dot(H), (x_max - x_min, y_max - y_min)
    )
    output_img[
        translation_dist[1] : rows1 + translation_dist[1],
        translation_dist[0] : cols1 + translation_dist[0],
    ] = img1

    # ----------------------------------------------------------------------
    # image, where overlay is 50/50 in area, where both images are
    warp_image = cv2.warpPerspective(
        img2, H_translation.dot(H), (x_max - x_min, y_max - y_min)
    )
    warp_mask = warp_image[:, :, 0] < 0.01

    black_img = np.zeros_like(warp_image)
    black_img[
        translation_dist[1] : rows1 + translation_dist[1],
        translation_dist[0] : cols1 + translation_dist[0],
    ] = img1
    unwarp_mask = black_img[:, :, 0] == 0

    output_img2_faded = cv2.addWeighted(warp_image, 0.5, black_img, 0.5, 0)

    combi_mask = warp_mask | unwarp_mask

    # masked faded
    output_img2_faded[combi_mask] = 0

    # mask original
    output_img2_outer = output_img.copy()
    output_img2_outer[~combi_mask] = 0

    output_img2 = output_img2_faded + output_img2_outer

    return output_img, output_img2


# Load our images
img2 = cv2.imread("./data/lr/img/second/0.jpg")
img1 = cv2.imread("./data/lr/img/first/0.jpg")

# img2 = cv2.imread("cutschlechtAchtern.jpg")
# img1 = cv2.imread("cutschlechtVorne.jpg")

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# Create our ORB detector and detect keypoints and descriptors
orb = cv2.ORB_create(nfeatures=16000)

# Find the key points and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

print(len(keypoints1))
featureimg = draw_features(img1, keypoints1[:60])
cv2.imwrite("test/temp_features.jpg", featureimg)

# Create a BFMatcher object.
# It will find all of the matching keypoints on two images
bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

# Find matching points
matches = bf.knnMatch(descriptors1, descriptors2, k=2)


all_matches = []
for m, n in matches:
    all_matches.append(m)


img3 = draw_matches(img1, keypoints1, img2, keypoints2, all_matches[:30])
cv2.imwrite("test/temp.jpg", img3)

# Finding the best matches
c = 0.75
good = []
for m, n in matches:
    if m.distance < c * n.distance:
        good.append(m)

img4 = draw_matches(img1, keypoints1, img2, keypoints2, good)
cv2.imwrite("test/temp_good.jpg", img4)


# Set minimum match condition
MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    # Convert keypoints to an argument for findHomography
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Establish a homography

    # Direction 1:
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(f"Matrix: {M}")
    result, result_faded = warpImages(img2, img1, M)
    cv2.imwrite("test/stitching_result_1_affine.jpg", result)
    cv2.imwrite("test/stitching_result_1_faded_affine.jpg", result_faded)

    # Direction 2:
    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    result, result_faded = warpImages(img1, img2, M)  # np.linalg.inv(M))
    cv2.imwrite("test/stitching_result_2_affine.jpg", result)
    cv2.imwrite("test/stitching_result_2_faded_affine.jpg", result_faded)
