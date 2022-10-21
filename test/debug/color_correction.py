import cv2
import numpy as np


class ColorCorrection:
    """Preprocessor which rotates images"""

    sift_features: int
    hws: float
    blur_size: float
    poly_degrees: tuple[int, int, int]

    def __init__(self) -> None:
        self.sift_features = 10000
        self.hws = 30
        self.blur_size = 11
        self.poly_degrees = (7, 1, 1)

    def process(self, img1, img2):
        kpoints1, kpoints2 = self._find_relevant_points(img1, img2)
        kpoints1, kpoints2 = self._non_max_suppression(kpoints1, kpoints2)
        color_factors = self._calculate_color_correction(img1, img2, kpoints1, kpoints2)
        color_factors, Y = self._interpolate_missing_factors(color_factors)
        img2_modified = self._apply_color_correction(img2, color_factors, Y)

        return img1, img2_modified

    def _find_relevant_points(self, img1, img2):
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create(nfeatures=self.sift_features)
        kpts_img1, descs_img1 = sift.detectAndCompute(img1, None)
        kpts_img2, descs_img2 = sift.detectAndCompute(img2, None)
        bfm = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)
        matches = bfm.match(descs_img1, descs_img2)

        # kpts1_img = cv2.drawKeypoints(img1, kpts_img1, None, color=(72, 144, 233))
        # cv2.imshow("f", kpts1_img)
        # cv2.waitKey(0)

        # matches_img = cv2.drawMatches(
        #     img1, kpts_img1, img2, kpts_img2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        # )
        # cv2.imshow("f", matches_img)
        # cv2.waitKey(0)

        matched1 = np.array([kpts_img1[m.queryIdx] for m in matches])
        matched2 = np.array([kpts_img2[m.trainIdx] for m in matches])
        return matched1, matched2

    def _non_max_suppression(self, points1, points2):
        valid = []

        for i in range(len(points1)):
            point_i = points1[i].pt
            is_max = True
            for j in range(len(valid)):
                point_j = points1[valid[j]].pt
                dist = (point_i[0] - point_j[0]) ** 2 + (point_i[1] - point_j[1]) ** 2
                if dist < self.hws**2:
                    is_max = False
                    break

            if is_max and points1[i].size < 5:
                valid.append(i)

        return points1[valid], points2[valid]

    def _calculate_color_correction(self, img1, img2, features1, features2):
        color_factors = np.zeros(shape=(256, 3), dtype=np.float32)
        factor_counter = np.zeros(shape=(256, 3), dtype=np.float32)

        img1_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
        img2_ycrcb = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)

        img1_ycrcb[:, :, 0] = img1_ycrcb[:, :, 0] / 255 * (235 - 16) + 16
        img1_ycrcb[:, :, 1] = img1_ycrcb[:, :, 1] / 255 * (240 - 16) + 16
        img1_ycrcb[:, :, 2] = img1_ycrcb[:, :, 2] / 255 * (240 - 16) + 16
        img2_ycrcb[:, :, 0] = img2_ycrcb[:, :, 0] / 255 * (235 - 16) + 16
        img2_ycrcb[:, :, 1] = img2_ycrcb[:, :, 1] / 255 * (240 - 16) + 16
        img2_ycrcb[:, :, 2] = img2_ycrcb[:, :, 2] / 255 * (240 - 16) + 16

        img_1_ycrcb_smooth = cv2.GaussianBlur(img1_ycrcb, (self.blur_size, self.blur_size), sigmaX=0.5, sigmaY=0.5)
        img_2_ycrcb_smooth = cv2.GaussianBlur(img2_ycrcb, (self.blur_size, self.blur_size), sigmaX=0.5, sigmaY=0.5)
        height1, width1 = img_1_ycrcb_smooth.shape[:2]
        height2, width2 = img_2_ycrcb_smooth.shape[:2]

        for f1, f2 in zip(features1, features2):
            p1, p2 = (round(f1.pt[0]), round(f1.pt[1])), (round(f2.pt[0]), round(f2.pt[1]))

            hws_1 = min(p1[0], width1 - p1[0] - 1, p1[1], height1 - p1[1] - 1)
            hws_2 = min(p2[0], width2 - p2[0] - 1, p2[1], height2 - p2[1] - 1)
            curr_hws = min(self.hws, hws_1, hws_2)

            img_1_section = img_1_ycrcb_smooth[
                p1[1] - curr_hws : p1[1] + curr_hws + 1, p1[0] - curr_hws : p1[0] + curr_hws + 1, :
            ]
            img_2_section = img_2_ycrcb_smooth[
                p2[1] - curr_hws : p2[1] + curr_hws + 1, p2[0] - curr_hws : p2[0] + curr_hws + 1, :
            ]

            color_diff = img_1_section / img_2_section

            for row in range(2 * curr_hws + 1):
                for col in range(2 * curr_hws + 1):
                    idx = min(234, max(15, round(img_2_section[row, col, 0])))
                    color_factors[idx, 0] = color_factors[idx, 0] + color_diff[row, col, 0]
                    factor_counter[idx, 0] = factor_counter[idx, 0] + 1

                    for chan in [1, 2]:
                        idx = min(239, max(15, round(img_2_section[row, col, chan])))
                        color_factors[idx, chan] = color_factors[idx, chan] + color_diff[row, col, chan]
                        factor_counter[idx, chan] = factor_counter[idx, chan] + 1

        color_factors /= factor_counter
        return color_factors

    def _interpolate_missing_factors(self, color_factors):
        color_factors = np.nan_to_num(color_factors)
        _, Y = np.meshgrid(range(3), range(256))

        p_y = np.polyfit(Y[color_factors[:, 0] > 0, 0], color_factors[color_factors[:, 0] > 0, 0], self.poly_degrees[0])
        p_cr = np.polyfit(
            Y[color_factors[:, 1] > 0, 0], color_factors[color_factors[:, 1] > 0, 1], self.poly_degrees[1]
        )
        p_cb = np.polyfit(
            Y[color_factors[:, 2] > 0, 0], color_factors[color_factors[:, 2] > 0, 2], self.poly_degrees[2]
        )

        range_y = list(range(16, 236))
        range_cr_cb = list(range(16, 241))
        color_factors[range_y, 0] = np.polyval(p_y, range_y)
        color_factors[range_cr_cb, 1] = np.polyval(p_cr, range_cr_cb)
        color_factors[range_cr_cb, 2] = np.polyval(p_cb, range_cr_cb)

        return color_factors, Y

    def _apply_color_correction(self, img2, color_factors, Y):
        color_maps = np.clip(np.around(color_factors[:, :] * Y), 0, 255)

        img2_ycrcb = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
        img2_ycrcb[:, :, 0] = img2_ycrcb[:, :, 0] / 255 * (235 - 16) + 16
        img2_ycrcb[:, :, 1] = img2_ycrcb[:, :, 1] / 255 * (240 - 16) + 16
        img2_ycrcb[:, :, 2] = img2_ycrcb[:, :, 2] / 255 * (240 - 16) + 16

        img_2_new = np.zeros_like(img2_ycrcb)
        img_2_new[:, :, 0] = color_maps[img2_ycrcb[:, :, 0], 0]
        img_2_new[:, :, 1] = color_maps[img2_ycrcb[:, :, 1], 1]
        img_2_new[:, :, 2] = color_maps[img2_ycrcb[:, :, 2], 2]

        img_2_new[:, :, 0] = (img_2_new[:, :, 0] - 16) / (235 - 16) * 255
        img_2_new[:, :, 1] = (img_2_new[:, :, 1] - 16) / (240 - 16) * 255
        img_2_new[:, :, 2] = (img_2_new[:, :, 2] - 16) / (240 - 16) * 255

        img_2_new_bgr = cv2.cvtColor(img_2_new.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        return img_2_new_bgr


img1 = cv2.imread(r"D:\UserFolders\Downloads\Translated.jpg")
img2 = cv2.imread(r"D:\UserFolders\Downloads\Warped_Overtune.jpg")

cc = ColorCorrection()
# img1_cc, img2_cc = cc.process(img1, img2)

color_maps = np.loadtxt("test/debug/color_maps.txt", delimiter=",", dtype=np.uint8)
color_maps -= 1
img2_ycrcb = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
img2_ycrcb[:, :, 0] = img2_ycrcb[:, :, 0] / 255 * (235 - 16) + 16
img2_ycrcb[:, :, 1] = img2_ycrcb[:, :, 1] / 255 * (240 - 16) + 16
img2_ycrcb[:, :, 2] = img2_ycrcb[:, :, 2] / 255 * (240 - 16) + 16

img_2_new = np.zeros_like(img2_ycrcb)
img_2_new[:, :, 0] = color_maps[img2_ycrcb[:, :, 0], 0]
img_2_new[:, :, 1] = color_maps[img2_ycrcb[:, :, 1], 2]
img_2_new[:, :, 2] = color_maps[img2_ycrcb[:, :, 2], 1]

img_2_new[:, :, 0] = (img_2_new[:, :, 0] - 16) / (235 - 16) * 255
img_2_new[:, :, 1] = (img_2_new[:, :, 1] - 16) / (240 - 16) * 255
img_2_new[:, :, 2] = (img_2_new[:, :, 2] - 16) / (240 - 16) * 255

img2_cc = cv2.cvtColor(img_2_new.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


cv2.imshow("t", img2_cc)
cv2.waitKey(0)

# A = np.array([2, 4, 6, 8, 3, 5, 7, 9]).reshape(2, 4)
# B = np.ones_like(A) * 10
# B[0, 0] = 0
# print(A / B)
