import cv2
import math
import imutils
import numpy as np


def get_quadrilateral_corner_degrees(coords):
    # Extract coordinates from the nested list
    x1, y1 = coords[0][0]
    x2, y2 = coords[1][0]
    x3, y3 = coords[2][0]
    x4, y4 = coords[3][0]

    # Calculate the corner angles in radians
    angle1 = math.degrees(math.atan2(y1, x1))
    angle2 = math.degrees(math.atan2(y2, x2))
    angle3 = math.degrees(math.atan2(y3, x3))
    angle4 = math.degrees(math.atan2(y4, x4))

    # Adjust angles to be positive and within the range [0, 360)
    angle1 = (angle1 + 360) % 360
    angle2 = (angle2 + 360) % 360
    angle3 = (angle3 + 360) % 360
    angle4 = (angle4 + 360) % 360

    # Return the corner angles
    return (angle1, angle2, angle3, angle4)


def get_triangle_corner_degrees(coords):
    # Extract coordinates from the nested list
    x1, y1 = coords[0][0]
    x2, y2 = coords[1][0]
    x3, y3 = coords[2][0]

    # Calculate the corner angles in radians
    angle1 = math.degrees(math.atan2(y1, x1))
    angle2 = math.degrees(math.atan2(y2, x2))
    angle3 = math.degrees(math.atan2(y3, x3))

    # Adjust angles to be positive and within the range [0, 360)
    angle1 = (angle1 + 360) % 360
    angle2 = (angle2 + 360) % 360
    angle3 = (angle3 + 360) % 360

    # Return the corner angles
    return (angle1, angle2, angle3)


def isImageContainCavetInRightDirection(im=None):
    '''

    '''
    height, width, _ = im.shape
    if height > width:
        return False
    # Resize:
    im = imutils.resize(im, width=640)
    im_cp = im.copy()
    # Padding:
    im_cp = cv2.copyMakeBorder(
        im_cp, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # Convert to grayscale:
    im_cp = cv2.cvtColor(im_cp, cv2.COLOR_BGR2GRAY)
    # Lower the brightness:
    # Alpha: Contrast control (1.0-3.0) - Beta: Brightness control (0-100)
    # Use mean of pixel value to calculate alpha and beta
    alpha = np.mean(im_cp) / 127.5
    beta = 100 - alpha * 40

    im_cp = cv2.convertScaleAbs(im_cp, alpha=alpha, beta=beta)
    equalized = cv2.equalizeHist(im_cp)
    # Morphological operations:
    kernel = np.ones((8, 8), np.uint8)
    dilation = cv2.dilate(equalized, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    # Thresholding:
    ret, thresh = cv2.threshold(
        opening, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # The edge will be the gradient black to white:
    edge = cv2.Canny(thresh, 60, 150, apertureSize=3)

    # Find contours:
    contours, hierarchy = cv2.findContours(
        edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Largest contour:
    largest_contour = max(contours, key=cv2.contourArea)
    # Convex hull:
    hull = cv2.convexHull(largest_contour)
    # Create a mask:
    mask = np.zeros(im_cp.shape, np.uint8)
    # Draw the contour on the mask:
    cv2.drawContours(mask, [hull], -1, 255, -1)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)

    # Find corners:
    corners = cv2.goodFeaturesToTrack(mask, 4, 0.5, 50)
    corners = np.intp(corners)
    # Draw the corners:
    # for i in corners:
    #     x, y = i.ravel()
    #     cv2.circle(im_cp, (x, y), 3, 255, -1)
    # cv2.imshow("im_cp", im_cp)
    # cv2.waitKey(0)
    # Check if the cornors equal to 4 and the angle between them is approximately 90 degree:
    # print(f"corners are: {corners}")
    if len(corners) == 4:
        # Check the angle:
        # Get the angles:
        angle1, angle2, angle3, angle4 = get_quadrilateral_corner_degrees(
            corners)
        # If have a least 1 angle is approximately 90 degree: -> rectangle
        if (angle1 >= 80 and angle1 <= 100) or (angle2 >= 80 and angle2 <= 100) or (angle3 >= 80 and angle3 <= 100) or (angle4 >= 80 and angle4 <= 100):
            return True
        else:
            return False
    elif len(corners) == 3:
        # Check the angle:
        # Get the angles:
        angle1, angle2, angle3 = get_triangle_corner_degrees(corners)
        # If have a least 1 angle is approximately 90 degree: -> rectangle
        if (angle1 >= 80 and angle1 <= 100) or (angle2 >= 80 and angle2 <= 100) or (angle3 >= 80 and angle3 <= 100):
            return True
        else:
            return False
    elif len(corners) < 3:
        return True
    else:
        return False
