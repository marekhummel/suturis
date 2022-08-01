# import the necessary packages
import numpy as np
import cv2


class VideoStitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X and initialize the
        # cached homography matrix
        self.cachedH = None
        self.ratio = 0.75
        self.reprojThresh = 4.0
        self.restrictions = ((0.1, 0.8), (0.25, 0.75))
        self.keypoints = None

    def stitch(self, images):
        imageA, imageB = images

        # if the cached homography matrix is None, then we need to
        # apply keypoint matching to construct it
        if self.cachedH is None:
            # detect keypoints and extract
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)
            # match features between the two images
            M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB)
            # if the match is None, then there aren't enough matched
            # keypoints to create a panorama
            if M is None:
                return None
            # cache the homography matrix
            self.cachedH = M
            self.keypoints = (kpsA, kpsB)

        # apply a perspective transform to stitch the images together
        # using the cached homography matrix
        result = cv2.warpPerspective(
            imageA,
            self.cachedH[1],
            (imageA.shape[1] + imageB.shape[1], imageA.shape[0]),
        )

        imageB_pad = np.pad(
            imageB,
            [(0, 0), (0, imageA.shape[1]), (0, 0)],
            mode="constant",
            constant_values=0,
        )
        result = cv2.addWeighted(result, 0.5, imageB_pad, 0.5, 0)

        resized_result = np.delete(
            result, np.s_[(int)(result.shape[1] * 0.55) :], axis=1
        )
        return resized_result

    def detectAndDescribe(self, image):
        # detect and extract features from the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        mask = self.computeMask(gray)
        cropped = cv2.bitwise_and(gray, gray, mask=mask)

        (kps, features) = sift.detectAndCompute(cropped, None)

        # arrays
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return (kps, features)

    def computeMask(self, image):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        h, w = image.shape[:2]
        restX, restY = self.restrictions
        cv2.rectangle(
            mask,
            ((int)(w * restX[0]), (int)(h * restY[0])),
            ((int)(w * restX[1]), (int)(h * restY[1])),
            255,
            -1,
        )
        return mask

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, self.reprojThresh)
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)
        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        # return the visualization
        return vis
