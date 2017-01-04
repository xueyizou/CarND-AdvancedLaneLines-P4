import numpy as np
import cv2
from scipy import signal
import calib
import preprocess

from glob import glob
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

from moviepy.editor import VideoFileClip


def running_mean(img, vert_slices, wsize):
    """
    Computes the horizontal moving histogram of an image
    :param img: The binary image (ch = 2)
    :param vert_slices: Number of vertical slices
    :param wsize: The window size
    :return: The computed histograms
    """
    size = img.shape[0] / vert_slices
    result = np.zeros(shape=(vert_slices, img.shape[1] - 24), dtype=np.float)

    for i in np.arange(vert_slices):
        start = i * size
        end = (i + 1) * size
        vertical_mean = np.mean(img[start:end], axis=0)
        window_sum = np.cumsum(np.insert(vertical_mean, 0, 0))
        result[i, :] = (window_sum[wsize:] - window_sum[:-wsize]) / wsize

    return result


class PerspectiveTransformer():
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def transform(self, img):
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def inverse_transform(self, img):
        return cv2.warpPerspective(img, self.M_inv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


SRC = np.float32([
    (0, 720),
    (520, 470),
    (760, 470),
    (1280, 720)])

DST = np.float32([
    (0, 720),
    (0, 0),
    (1280, 0),
    (1280, 720)])


class LaneDetector():
    def __init__(self, perspective_src, perspective_dst, cam_calibration=None):
        self.cam_calibration = cam_calibration
        self.perspective_src = perspective_src
        self.perspective_dst = perspective_dst
        self.perspective_transformer = PerspectiveTransformer(perspective_src, perspective_dst)

    def process_frame(self, frame):
        resize = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

        # Apply the distortion correction to the raw image.
        if cam_calibration is not None:
            dist_correct = calib.undistort(resize, self.cam_calibration)
        else:
            dist_correct = np.copy(resize)

        # Filter the image
        blur = preprocess.guassian_blur(dist_correct, 5)

        # Use color transforms, gradients, etc., to create a thresholded binary image.
        binary = self.__threshold_image(blur)

        # Apply a perspective transform to rectify binary image ("birds-eye view").
        birdseye = self.perspective_transformer.transform(binary)
        birdseye = preprocess.median_blur(birdseye, 5)

        # Form a moving average and obtain the peaks
        moving_avg = running_mean(birdseye, 10, resize.shape[1]/50)

        # Find the lanes in each vertical slice
        lane_overlay = np.zeros_like(birdseye)
        for i, avg in enumerate(moving_avg):
            start = int(i * resize.shape[0]/10)
            end = int((i + 1) * resize.shape[0]/10)
            lane_mask = np.argwhere(avg > 0.05)
            #print ('Lane Mask: {}'.format(lane_mask))
            if len(lane_mask):
                lane_min = np.min(lane_mask)
                lane_max = np.max(lane_mask)
                #print(lane_min, end, lane_max, start)
                cv2.rectangle(lane_overlay, (lane_min, end), (lane_max, start), 255, thickness=10)
            #break



        fig = plt.figure()
        fig.add_subplot(331), plt.imshow(frame), plt.title('Original')
        fig.add_subplot(332), plt.imshow(resize), plt.title('Resized')
        fig.add_subplot(333), plt.imshow(dist_correct), plt.title('Undistorted')
        fig.add_subplot(334), plt.imshow(blur), plt.title('Filtered')
        fig.add_subplot(335), plt.imshow(binary, cmap='gray'), plt.title('Thresholded')
        fig.add_subplot(336), plt.imshow(birdseye, cmap='gray'), plt.title('Overhead')
        fig.add_subplot(337), plt.bar(np.arange(len(moving_avg[0])),np.mean(moving_avg, axis=0)), plt.title('Moving avg')
        fig.add_subplot(338), plt.imshow(lane_overlay, cmap='gray'), plt.title('Moving avg')
        #fig.add_subplot(338), plt.imshow(result), plt.title('Output')
        plt.show()

        return birdseye

    @staticmethod
    def __threshold_image(img):
        """
        Thresholds an image based on various criteria
        :param img: Image to be thresholded
        :return: Returns a thresholded image
        """
        # Compute color thresholds
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gamma = preprocess.gamma_threshold(img, 0.01)
        white = cv2.inRange(hsv, (20, 0, 180), (255, 80, 255))
        yellow = cv2.inRange(hsv, (0, 100, 100), (50, 255, 255))
        color = cv2.bitwise_or(cv2.bitwise_or(gamma, white), yellow)

        # Compute Sobel thresholds on the L channel
        sobel_x = preprocess.Sobel.absolute_thresh(hsv[:, :, 2], orientation='x', sobel_kernel=5, threshold=(50, 255))
        sobel_y = preprocess.Sobel.absolute_thresh(hsv[:, :, 2], orientation='y', sobel_kernel=5, threshold=(50, 255))
        sobel_l = np.copy(cv2.bitwise_or(sobel_x, sobel_y))

        # Compute Sobel thresholds on the S channel
        sobel_x = preprocess.Sobel.absolute_thresh(hsv[:, :, 2], orientation='x', sobel_kernel=5, threshold=(50, 255))
        sobel_y = preprocess.Sobel.absolute_thresh(hsv[:, :, 2], orientation='y', sobel_kernel=5, threshold=(50, 255))
        sobel_s = np.copy(cv2.bitwise_or(sobel_x, sobel_y))

        # Combine the Sobel and filter it
        sobel = preprocess.median_blur(cv2.bitwise_or(sobel_l, sobel_s), 13)

        # Combine all the thresholds and form a binary image
        result = np.zeros_like(sobel)
        result[(color >= .5) | (sobel >= .5)] = 1

        # fig = plt.figure()
        # fig.add_subplot(331), plt.imshow(img)
        # fig.add_subplot(332), plt.imshow(gamma, cmap='gray'), plt.title('Gamma')
        # fig.add_subplot(333), plt.imshow(white, cmap='gray'), plt.title('White')
        # fig.add_subplot(334), plt.imshow(yellow, cmap='gray'), plt.title('Yellow')
        # fig.add_subplot(335), plt.imshow(color, cmap='gray'), plt.title('Color')
        # fig.add_subplot(336), plt.imshow(sobel_l, cmap='gray'), plt.title('Sobel L')
        # fig.add_subplot(337), plt.imshow(sobel_s, cmap='gray'), plt.title('Sobel S')
        # fig.add_subplot(338), plt.imshow(sobel, cmap='gray'), plt.title('Sobel')
        # fig.add_subplot(339), plt.imshow(result, cmap='gray'), plt.title('Final')
        # plt.show()

        return result

if __name__ == "__main__":
    cam_calibration = calib.calibrate_camera('camera_cal', (9, 6), (720, 1280, 3))

    images = glob('test_images/*')

    for idx, img_path in enumerate(images):
        ld = LaneDetector(SRC, DST, cam_calibration=cam_calibration)
        img = imread(img_path)
        res = ld.process_frame(img)

    # ld = LaneDetector(SRC, DST, cam_calibration=cam_calibration)
    #
    # project_output = 'out.mp4'
    # clip1 = VideoFileClip('harder_challenge_video.mp4')
    # project_clip = clip1.fl_image(ld.process_frame)
    # project_clip.write_videofile(project_output, audio=False)