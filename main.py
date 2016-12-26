import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def calibrate_camera(calibration_dir, num_corners=(9, 6)):
    """
    Calibrates the camera based on a set of images
    :param calibration_dir: A directory where the calibration images are stored
    :param num_corners: Number of corners in each of the chessboard images
    :return: mtx, dist: The transformation matrix along with the distortion
    """
    # Preconditions
    assert os.path.exists(calibration_dir)

    # Constants
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Initialize the image and object points
    obj_points = []
    img_points = []
    gray_shape = None

    # Setup the object points
    objp = np.zeros((num_corners[0]*num_corners[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:num_corners[0],0:num_corners[1]].T.reshape(-1,2)

    # Get the list of images
    img_list = os.listdir(calibration_dir)

    # Loop through the images
    for img_path in img_list:

        # Read the image
        img = cv2.imread(os.path.join(calibration_dir, img_path))

        # Convert to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape

        # Find the cornerts
        ret, corners = cv2.findChessboardCorners(gray, num_corners, None)

        # If corners were found
        if ret:
            # Append the values
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)

    # Let's calibrate the camera with all the known points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_shape[::-1], None, None)

    # Return the matrix and the distrotion
    return mtx, dist


def sobel_threshold(img, ksize=3, abs_thresh=(64, 255), mag_thresh=(128, 255), dir_thres=(0.7, 1.2)):
    # Take the Sobel in both x and y direction
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

    # Take the magnitude
    abs_sobelx = np.absolute(sobel_x)
    abs_sobely = np.absolute(sobel_y)
    abs_sobel = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)

    # Compute the direction
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)

    # Scale it to 0 - 255
    scaled_abs_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    scaled_mag_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Create a binary image
    answer = np.zeros_like(scaled_abs_sobelx)
    answer[((scaled_abs_sobelx >= abs_thresh[0]) & (scaled_abs_sobelx <= abs_thresh[1])) |
           ((scaled_mag_sobel >= mag_thresh[0]) & (scaled_mag_sobel <= mag_thresh[1]) &
            (dir_sobel >= dir_thres[0]) & (dir_sobel <= dir_thres[1]))] = 255

    # fig = plt.figure()
    # fig.suptitle('Sobel')
    # fig.add_subplot(231), plt.imshow(img, cmap='gray'), plt.title('Sobel Input')
    # fig.add_subplot(232), plt.imshow(scaled_abs_sobelx, cmap='gray'), plt.title('Sobelx')
    # fig.add_subplot(234), plt.imshow(scaled_mag_sobel, cmap='gray'), plt.title('Sobel Magnitude')
    # fig.add_subplot(235), plt.imshow(dir_sobel, cmap='gray'), plt.title('Sobel Direction')
    # fig.add_subplot(236), plt.imshow(answer, cmap='gray'), plt.title('Final')
    # plt.show()

    # Return the binary image
    return answer


def process(img, mtx, dist):
    """
    Processes a single image
    :param img: The color image to be processed (should be in RBG)
    :param mtx: The calibration matrix
    :param dist: The clibration distortion
    :return: The processes image with the lane lines drawn on it
    """
    """
    CONSTANTS
    """
    SAT_THRESHOLD_MIN = 128
    SAT_THRESHOLD_MAX = 255
    GUASSIAN_KERNEL = 15
    CANNY_THRESHOLD_LOW = 30
    CANNY_THRESHOLD_HIGH = 40

    """
    PIPELINE
    """
    # Make a copy
    img = np.copy(img)

    # Undistort the image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Aply a guassian blur
    blur = cv2.GaussianBlur(undist, (GUASSIAN_KERNEL, GUASSIAN_KERNEL), 0)

    # Convert to HLS
    hls = cv2.cvtColor(blur, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Canny
    canny = cv2.Canny(l_channel, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
    canny_binary = np.zeros_like(canny)
    canny_binary[canny > 128] = 255.

    # Sobel x and threshold the gradient
    sobel_binary = sobel_threshold(l_channel)

    # Threshold saturation channel
    sat_binary = np.zeros_like(s_channel)
    sat_binary[(s_channel >= SAT_THRESHOLD_MIN) & (s_channel <= SAT_THRESHOLD_MAX)] = 255.

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((sobel_binary, canny_binary, sat_binary))

    # Debug
    fig = plt.figure()
    fig.add_subplot(3, 3, 1), plt.imshow(img), plt.title('Original')
    fig.add_subplot(3, 3, 2), plt.imshow(undist), plt.title('Undistorted')
    fig.add_subplot(3, 3, 3), plt.imshow(blur), plt.title('Filtered')
    #fig.add_subplot(3, 3, 4), plt.imshow(sobel, cmap='gray'), plt.title('Sobel X')
    fig.add_subplot(3, 3, 5), plt.imshow(sobel_binary, cmap='gray'), plt.title('Sobel Binary')
    fig.add_subplot(3, 3, 6), plt.imshow(s_channel, cmap='gray'), plt.title('Saturation')
    fig.add_subplot(3, 3, 7), plt.imshow(sat_binary, cmap='gray'), plt.title('Saturation Binary')
    fig.add_subplot(3, 3, 8), plt.imshow(canny_binary, cmap='gray'), plt.title('Canny')
    fig.add_subplot(3, 3, 9), plt.imshow(color_binary), plt.title('Final Binary')
    plt.show()

if __name__ == "__main__":
    mtx, dist = calibrate_camera('camera_cal')

    test_imgs = os.listdir('test_images')
    for img_path in test_imgs:
        img = cv2.imread(os.path.join('test_images', img_path))
        process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mtx, dist)
