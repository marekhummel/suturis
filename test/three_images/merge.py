import numpy as np
import cv2


def compute_homography(img1, img2, kpts1, kpts2):
    # kpts_img1 = cv2.KeyPoint_convert(kpts1)
    # kpts_img2 = cv2.KeyPoint_convert(kpts2)
    # cv2.imshow("t1", cv2.drawKeypoints(img1, kpts_img1, None, color=(0, 0, 0)))
    # cv2.imshow("t2", cv2.drawKeypoints(img2, kpts_img2, None, color=(0, 0, 0)))

    dst_pts = np.array(kpts1, dtype=np.float32).reshape(-1, 1, 2)
    src_pts = np.array(kpts2, dtype=np.float32).reshape(-1, 1, 2)

    # Establish a homography
    homography, _ = cv2.findHomography(src_pts, dst_pts, 0)
    return homography


def compute_canvas(img_shape, homography1, homography2):
    height, width = img_shape[:2]
    corners_basic = [[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]
    corners = np.array(corners_basic, dtype=np.float32).reshape(4, 1, 2)

    # Transform second image corners with homography
    corners_homography1 = cv2.perspectiveTransform(corners, homography1)
    corners_homography2 = cv2.perspectiveTransform(corners, homography2)

    # Find min and max of all corners
    all_corners = np.concatenate((corners, corners_homography1, corners_homography2), axis=0)
    x_min, y_min = np.around(all_corners.min(axis=0).ravel()).astype(np.int32)
    x_max, y_max = np.around(all_corners.max(axis=0).ravel()).astype(np.int32)

    # Compute translation and canvas size
    translation = (-x_min, -y_min)
    canvas_size = (x_max - x_min + 1, y_max - y_min + 1)

    return translation, canvas_size


def apply_warping(img1, img2, img3, homography1, homography2, canvas_size, translation):
    target_width, target_height = canvas_size
    tx, ty = translation

    img1_height, img1_width = img1.shape[:2]
    img1_tf = np.zeros(shape=(target_height, target_width, 3), dtype=np.uint8)
    img1_tf[ty : img1_height + ty, tx : img1_width + tx] = img1

    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    img2_tf = cv2.warpPerspective(img2, translation_matrix @ homography1, canvas_size)
    img3_tf = cv2.warpPerspective(img3, translation_matrix @ homography2, canvas_size)

    return img1_tf, img2_tf, img3_tf


# ---------------------------------------------


def mask(base_img, pts_base, query_img1, pts_query1, query_img2, pts_query2):

    homography1 = compute_homography(base_img, query_img1, pts_base, pts_query1)
    homography2 = compute_homography(base_img, query_img2, pts_base, pts_query2)
    translation, canvas_size = compute_canvas(img1.shape, homography1, homography2)
    img1_tf, img2_tf, img3_tf = apply_warping(
        base_img, query_img1, query_img2, homography1, homography2, canvas_size, translation
    )

    mask = np.full_like(img1_tf, 1 / 3, dtype=np.float64)
    img1_masked = img1_tf.astype(np.float64) * mask
    img2_masked = img2_tf.astype(np.float64) * mask
    img3_masked = img3_tf.astype(np.float64) * mask
    final = (img1_masked + img2_masked + img3_masked).astype(np.uint8)

    return final


if __name__ == "__main__":
    img1 = cv2.imread("test/three_images/img1.jpg").astype(np.uint8)
    img2 = cv2.imread("test/three_images/img2.jpg").astype(np.uint8)
    img3 = cv2.imread("test/three_images/img3.jpg").astype(np.uint8)

    pts1 = [[895, 345], [892, 368], [706, 276], [713, 418], [562, 277], [562, 407]]
    pts2 = [[841, 380], [837, 402], [659, 310], [661, 447], [518, 305], [514, 432]]
    pts3 = [[802, 318], [803, 295], [988, 393], [1009, 248], [1116, 397], [1127, 268]]

    final = mask(img3, pts3, img1, pts1, img2, pts2)
    final = cv2.resize(final, (1280, 720))
    cv2.imshow("f", final)
    cv2.waitKey(0)
