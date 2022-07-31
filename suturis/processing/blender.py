import numpy as np
import cv2
from matplotlib import pyplot as plt


mask1 = cv2.threshold(cv2.cvtColor(cv2.imread('./data/mask/left.jpg'), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]
mask2 = cv2.threshold(cv2.cvtColor(cv2.imread('./data/mask/right.jpg'), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]


async def blend(image1, image2):
    return (cv2.bitwise_and(image1, image1, mask=mask1), cv2.bitwise_and(image2, image2, mask=mask2))


async def blend_wild(image1, image2):
    imgL = cv2.pyrDown(image1)  # downscale images for faster processing
    imgR = cv2.pyrDown(image2)

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=16,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )

    print("computing disparity...")
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print(
        "generating 3d point cloud...",
    )
    h, w = imgL.shape[:2]
    f = 0.8 * w  # guess for focal length
    Q = np.float32(
        [
            [1, 0, 0, -0.5 * w],
            [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
            [0, 0, 0, -f],  # so that y-axis looks up
            [0, 0, 1, 0],
        ]
    )
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = "out.ply"
    # write_ply(out_fn, out_points, out_colors)
    print("%s saved" % out_fn)

    return (disp - min_disp) / num_disp


async def blend_SGBM(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(numDisparities=48, minDisparity=16, blockSize=15)
    disparity = stereo.compute(gray1, gray2)

    displayable = disparity.astype(np.float32) / 16.0
    # plt.imshow(disparity, "gray")
    # plt.show()

    return (displayable - 16) / 48
