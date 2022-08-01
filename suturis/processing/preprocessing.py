import cv2


mask1 = cv2.threshold(cv2.cvtColor(cv2.imread('./data/mask/left.jpg'), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]
mask2 = cv2.threshold(cv2.cvtColor(cv2.imread('./data/mask/right.jpg'), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1]


def preprocess(*images):
    image1, image2 = images
    return _mask_out(image1, image2)


def _mask_out(image1, image2):
    return (cv2.bitwise_and(image1, image1, mask=mask1), cv2.bitwise_and(image2, image2, mask=mask2))
