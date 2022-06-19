import cv2
import numpy as np

stitcher = cv2.Stitcher_create()


async def compute(image1, image2):
    stitched = _current_stitch(image1, image2)
    return _combine_with_input(image1, image2, stitched)


def _stitch(image1, image2):
    success, image = stitcher.stitch(image1, image2)
    return image


def _test_stitch(image1, image2):
    stitched = cv2.rotate(image2, rotateCode=cv2.ROTATE_180)
    return stitched


def _current_stitch(image1, image2):
    def warpImages(img1, img2, H):
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]

        list_of_points_1 = np.float32(
            [[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]
        ).reshape(-1, 1, 2)
        temp_points = np.float32(
            [[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]
        ).reshape(-1, 1, 2)

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
            translation_dist[1]: rows1 + translation_dist[1],
            translation_dist[0]: cols1 + translation_dist[0],
        ] = img1

        # ----------------------------------------------------------------------
        # image, where overlay is 50/50 in area, where both images are
        warp_image = cv2.warpPerspective(
            img2, H_translation.dot(H), (x_max - x_min, y_max - y_min)
        )
        warp_mask = warp_image[:, :, 0] < 0.01

        black_img = np.zeros_like(warp_image)
        black_img[
            translation_dist[1]: rows1 + translation_dist[1],
            translation_dist[0]: cols1 + translation_dist[0],
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

    # Create our ORB detector and detect keypoints and descriptors
    orb = cv2.ORB_create(nfeatures=16000)

    # Find the key points and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # Find matching points
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Convert keypoints to an argument for findHomography
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Establish a homography (src to dst)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    result, result_faded = warpImages(image2, image1, M)
    return result_faded


def _combine_with_input(image1, image2, stitch):
    image1 = cv2.resize(image1, (0, 0), None, 0.5, 0.5)
    image2 = cv2.resize(image2, (0, 0), None, 0.5, 0.5)
    h, w = image1.shape[:2]
    stitch = cv2.resize(stitch, (w, h), interpolation=cv2.INTER_CUBIC)
    stitch = np.pad(
        stitch, [(0, 0), (w // 2, w // 2), (0, 0)], mode="constant", constant_values=0
    )
    image = np.concatenate((image1, image2), axis=1)
    image = np.concatenate((image, stitch), axis=0)
    return image
