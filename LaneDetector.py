import numpy as np
import cv2
from scipy import signal
import calib
import preprocess

from glob import glob
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

from moviepy.editor import VideoFileClip


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


def histogram_lane_detection(img, steps, search_window, h_window, v_window):
    all_x = []
    all_y = []
    masked_img = img[:, search_window[0]:search_window[1]]
    histograms = np.zeros((steps, masked_img.shape[1]))
    pixels_per_step = img.shape[0] // steps

    for i in range(steps):
        start = masked_img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step
        histogram = np.sum(masked_img[end:start, :], axis=0)
        histograms[i] = histogram

    histograms = histogram_smoothing(histograms, window=v_window)

    for i, histogram in enumerate(histograms):
        start = masked_img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step

        histogram_smooth = signal.medfilt(histogram, h_window)
        peaks = np.array(signal.find_peaks_cwt(histogram_smooth, np.arange(1, 50)))

        highest_peak = detect_highest_peak_in_area(histogram_smooth, peaks, threshold=1000)
        if highest_peak is not None:
            center = (start + end) // 2
            x, y = get_pixel_in_window(masked_img, highest_peak, center, pixels_per_step)

            all_x.extend(x)
            all_y.extend(y)

    all_x = np.array(all_x) + search_window[0]
    all_y = np.array(all_y)

    return all_x, all_y


def highest_n_peaks(histogram, peaks, n=2, threshold=0):
    if len(peaks) == 0:
        return []

    peak_list = []
    for peak in peaks:
        y = histogram[peak]
        if y > threshold:
            peak_list.append((peak, histogram[peak]))
    peak_list = sorted(peak_list, key=lambda x: x[1], reverse=True)

    if len(peak_list) == 0:
        return []
    else:
        x, y = zip(*peak_list)
        return x[:n]


def histogram_smoothing(histograms, window=3):
    smoothed = np.zeros_like(histograms)
    for h_i, hist in enumerate(histograms):
        window_sum = np.zeros_like(hist)
        for w_i in range(window):
            index = w_i + h_i - window // 2
            if index < 0:
                index = 0
            elif index > len(histograms) - 1:
                index = len(histograms) - 1

            window_sum += histograms[index]

        smoothed[h_i] = window_sum / window

    return smoothed


def detect_highest_peak_in_area(histogram, peaks, threshold=0):
    peak = highest_n_peaks(histogram, peaks, n=1, threshold=threshold)
    if len(peak) == 1:
        return peak[0]
    else:
        return None


def detect_lane_along_poly(img, poly, steps):
    pixels_per_step = img.shape[0] // steps
    all_x = []
    all_y = []

    for i in range(steps):
        start = img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step

        center = (start + end) // 2
        x = poly(center)

        x, y = get_pixel_in_window(img, x, center, pixels_per_step)

        all_x.extend(x)
        all_y.extend(y)

    return all_x, all_y


def get_pixel_in_window(img, x_center, y_center, size):
    half_size = size // 2
    window = img[y_center - half_size:y_center + half_size,
             x_center - half_size:x_center + half_size]

    x, y = (window.T == 255).nonzero()

    x = x + x_center - half_size
    y = y + y_center - half_size

    return x, y


def calculate_lane_area(lanes, img_height, steps):
    """
    Expects the line polynom to be a function of y.
    """
    points_left = np.zeros((steps + 1, 2))
    points_right = np.zeros((steps + 1, 2))

    for i in range(steps + 1):
        pixels_per_step = img_height // steps
        start = img_height - i * pixels_per_step

        points_left[i] = [lanes[0].best_fit_poly(start), start]
        points_right[i] = [lanes[1].best_fit_poly(start), start]

    return np.concatenate((points_left, points_right[::-1]), axis=0)


def are_lanes_plausible(lane_one, lane_two, parall_thres=(0.0002, 0.5), dist_thres=(450, 550)):
    is_parall = lane_one.is_current_fit_parallel(lane_two, threshold=parall_thres)
    dist = lane_one.get_current_fit_distance(lane_two)
    is_plausible_dist = dist_thres[0] < dist < dist_thres[1]
    return is_parall & is_plausible_dist


def draw_poly(img, poly, steps, color, thickness=10, dashed=False):
    img_height = img.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start)), start)
        end_point = (int(poly(end)), end)

        if dashed == False or i % 2 == 1:
            img = cv2.line(img, end_point, start_point, color, thickness)

    return img


def draw_poly_arr(img, poly, steps, color, thickness=10, dashed=False, tipLength=1):
    img_height = img.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start)), start)
        end_point = (int(poly(end)), end)

        if dashed == False or i % 2 == 1:
            img = cv2.arrowedLine(img, end_point, start_point, color, thickness, tipLength=tipLength)

    return img


def outlier_removal(x, y, q=10):
    x = np.array(x)
    y = np.array(y)

    lower_bound = np.percentile(x, q)
    upper_bound = np.percentile(x, 100 - q)
    selection = (x >= lower_bound) & (x <= upper_bound)
    return x[selection], y[selection]


def calc_curvature(fit_cr):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension

    y = np.array(np.linspace(0, 719, num=10))
    x = np.array([fit_cr(x) for x in y])
    y_eval = np.max(y)

    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)

    curverad = ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

    return curverad


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, n_frames=1, x=None, y=None):
        # Frame memory
        self.n_frames = n_frames
        # was the line detected in the last iteration?
        self.detected = False
        # number of pixels added per frame
        self.n_pixel_per_frame = []
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = None
        # Polynom for the current coefficients
        self.current_fit_poly = None
        # Polynom for the average coefficients over the last n iterations
        self.best_fit_poly = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        if x is not None:
            self.update(x, y)

    def update(self, x, y):
        assert len(x) == len(y), 'x and y have to be the same size'

        self.allx = x
        self.ally = y

        self.n_pixel_per_frame.append(len(self.allx))
        self.recent_xfitted.extend(self.allx)

        if len(self.n_pixel_per_frame) > self.n_frames:
            n_x_to_remove = self.n_pixel_per_frame.pop(0)
            self.recent_xfitted = self.recent_xfitted[n_x_to_remove:]

        self.bestx = np.mean(self.recent_xfitted)

        self.current_fit = np.polyfit(self.allx, self.ally, 2)

        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = (self.best_fit * (self.n_frames - 1) + self.current_fit) / self.n_frames

        self.current_fit_poly = np.poly1d(self.current_fit)
        self.best_fit_poly = np.poly1d(self.best_fit)

    def is_current_fit_parallel(self, other_line, threshold=(0, 0)):
        first_coefi_dif = np.abs(self.current_fit[0] - other_line.current_fit[0])
        second_coefi_dif = np.abs(self.current_fit[1] - other_line.current_fit[1])
        is_parallel = first_coefi_dif < threshold[0] and second_coefi_dif < threshold[1]

        return is_parallel

    def get_current_fit_distance(self, other_line):
        return np.abs(self.current_fit_poly(719) - other_line.current_fit_poly(719))


HIST_STEPS = 10
OFFSET = 200
SRC = np.float32([
    (300, 720),
    (580, 470),
    (730, 470),
    (1100, 720)])

DST = np.float32([
    (SRC[0][0] - 50 + OFFSET, SRC[0][1]),
    (SRC[0][0] - 50 + OFFSET, 0),
    (SRC[-1][0] - OFFSET, 0),
    (SRC[-1][0] - OFFSET, SRC[0][1])])

FRAME_MEMORY = 5


class LaneDetector():
    def __init__(self, perspective_src, perspective_dst, img_size, n_frames=1, cam_calibration=None, line_segments=10):
        # Frame memory
        self.n_frames = n_frames
        self.cam_calibration = cam_calibration
        self.line_segments = line_segments
        self.img_size = img_size

        self.left_line = Line(n_frames)
        self.right_line = Line(n_frames)
        self.center_poly = None
        self.curvature = 0.0
        self.offset = 0.0

        self.perspective_src = perspective_src
        self.perspective_dst = perspective_dst
        self.perspective_transformer = PerspectiveTransformer(perspective_src, perspective_dst)

        self.n_frames_processed = 0

    def __line_found(self, left, right):
        if len(left[0]) == 0 or len(right[0]) == 0:
            return False
        else:
            left_x, left_y = outlier_removal(left[0], left[1])
            right_x, right_y = outlier_removal(right[0], right[1])
            new_left = Line(y=left_x, x=left_y)
            new_right = Line(y=right_x, x=right_y)
            return are_lanes_plausible(new_left, new_right)

    def __draw_info_panel(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Radius of Curvature = %d(m)' % self.curvature, (50, 50), font, 1, (255, 255, 255), 2)
        left_or_right = 'left' if self.offset < 0 else 'right'
        cv2.putText(img, 'Vehicle is %.2fm %s of center' % (np.abs(self.offset), left_or_right), (50, 100), font, 1,
                    (255, 255, 255), 2)

    def __draw_lane_overlay(self, img):
        overlay = np.zeros([*img.shape])
        mask = np.zeros([img.shape[0], img.shape[1]])

        lane_area = calculate_lane_area((self.left_line, self.right_line), img.shape[0], 20)
        mask = cv2.fillPoly(mask, np.int32([lane_area]), 255)
        mask = self.perspective_transformer.inverse_transform(mask)

        overlay[mask == 255] = (255, 128, 0)
        selection = (overlay != 0)
        img[selection] = img[selection] * 0.3 + overlay[selection] * 0.7

        center_line = np.zeros([img.shape[0], img.shape[1]])
        center_line = draw_poly_arr(center_line, self.center_poly, 20, 255, 5, True, tipLength=0.5)
        center_line = self.perspective_transformer.inverse_transform(center_line)
        img[center_line == 255] = (255, 64, 13)

        lines_best = np.zeros([img.shape[0], img.shape[1]])
        lines_best = draw_poly(lines_best, self.left_line.best_fit_poly, 5, 255)
        lines_best = draw_poly(lines_best, self.right_line.best_fit_poly, 5, 255)
        lines_best = self.perspective_transformer.inverse_transform(lines_best)
        img[lines_best == 255] = (255, 190, 13)

    def process_frame(self, frame):
        orig_frame = np.copy(frame)

        resize = imresize(frame, self.img_size)

        # Apply the distortion correction to the raw image.
        if cam_calibration is not None:
            dist_correct = calib.undistort(resize, self.cam_calibration)

        # Use color transforms, gradients, etc., to create a thresholded binary image.
        blur = cv2.GaussianBlur(dist_correct, (5, 5), 0)
        binary = self.__threshold_image(blur)

        # Apply a perspective transform to rectify binary image ("birds-eye view").
        birdseye = self.perspective_transformer.transform(binary)

        # mask outside are of persp trans
        birdseye[:, frame.shape[1] - OFFSET:] = 0
        birdseye[:, :OFFSET] = 0

        if self.n_frames_processed != 0:
            left_x, left_y = detect_lane_along_poly(birdseye, self.left_line.best_fit_poly, self.line_segments)
            right_x, right_y = detect_lane_along_poly(birdseye, self.right_line.best_fit_poly, self.line_segments)

            lines_detected = self.__line_found((left_x, left_y), (right_x, right_y))
        else:
            lines_detected = False

        if lines_detected == False:
            left_x, left_y = histogram_lane_detection(
                birdseye, self.line_segments, (OFFSET, frame.shape[1] // 2), h_window=21, v_window=3)
            right_x, right_y = histogram_lane_detection(
                birdseye, self.line_segments, (frame.shape[1] // 2, frame.shape[1] - OFFSET), h_window=21, v_window=3)

            lines_detected = self.__line_found((left_x, left_y), (right_x, right_y))

        if lines_detected == True or self.n_frames_processed == 0:
            try:
                # switch x and y since lines are almost vertical
                self.left_line.update(y=left_x, x=left_y)
                # switch x and y since lines are almost vertical
                self.right_line.update(y=right_x, x=right_y)

                self.center_poly = (self.left_line.best_fit_poly + self.right_line.best_fit_poly) / 2
                self.curvature = calc_curvature(self.center_poly)
                self.offset = (frame.shape[1] / 2 - self.center_poly(719)) * 3.7 / 700

                self.__draw_lane_overlay(orig_frame)
                self.__draw_info_panel(orig_frame)

                self.n_frames_processed += 1
            except:
                print('error while searching a line')

        fig = plt.figure()
        fig.add_subplot(331), plt.imshow(frame), plt.title('Original')
        fig.add_subplot(332), plt.imshow(resize), plt.title('Resized')
        fig.add_subplot(333), plt.imshow(dist_correct), plt.title('Undistorted')
        fig.add_subplot(334), plt.imshow(blur), plt.title('Filtered')
        fig.add_subplot(335), plt.imshow(binary, cmap='gray'), plt.title('Thresholded')
        fig.add_subplot(336), plt.imshow(birdseye, cmap='gray'), plt.title('Overhead')
        fig.add_subplot(338), plt.imshow(orig_frame), plt.title('Output')
        plt.show()

        return orig_frame

    @staticmethod
    def __threshold_image(img):
        """
        Thresholds an image based on various criteria
        :param img: Image to be thresholded
        :return: Returns a thresholded image
        """
        result = np.zeros(shape=(6, img.shape[0], img.shape[1]), dtype=np.uint8)

        # Compute color thresholds
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        result[0, :, :] = preprocess.gamma_threshold(img, 0.01)
        result[1, :, :] = cv2.inRange(hsv, (0, 0, 200), (255, 30, 255))
        result[2, :, :] = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))

        # Compute Sobel thresholds
        gradx = preprocess.Sobel.absolute_thresh(hsv[:, :, 2], orientation='x', sobel_kernel=11, threshold=(30, 255))
        grady = preprocess.Sobel.absolute_thresh(hsv[:, :, 2], orientation='y', sobel_kernel=11, threshold=(30, 255))
        x_and_y = (grady & gradx)

        mag_binary = preprocess.Sobel.magnitude_thresh(hsv[:, :, 2], sobel_kernel=11, threshold=(30, 255))
        dir_binary = preprocess.Sobel.direction_threshold(hsv[:, :, 2], sobel_kernel=5,
                                                          threshold=(np.pi / 5, np.pi / 2.3))

        mask = (x_and_y == 1.) | ((mag_binary * dir_binary) == 1.)
        result[3, :, :] = (mask * 255).astype(np.uint8)

        # Combine into a single threshold
        combined = np.max(result, axis=0)

        # fig = plt.figure()
        # fig.add_subplot(231), plt.imshow(img)
        # fig.add_subplot(232), plt.imshow(result[0, :, :], cmap='gray'), plt.title('Gamma')
        # fig.add_subplot(233), plt.imshow(result[1, :, :], cmap='gray'), plt.title('White')
        # fig.add_subplot(234), plt.imshow(result[2, :, :], cmap='gray'), plt.title('Yellow')
        # fig.add_subplot(235), plt.imshow(result[3, :, :], cmap='gray'), plt.title('Sobel')
        # fig.add_subplot(236), plt.imshow(combined, cmap='gray'), plt.title('Final')
        # plt.show()

        return combined

if __name__ == "__main__":
    cam_calibration = calib.calibrate_camera('camera_cal', (9, 6), (720, 1280, 3))

    images = glob('test_images/*')

    for idx, img_path in enumerate(images):
        ld = LaneDetector(SRC, DST, (720, 1280), n_frames=FRAME_MEMORY, cam_calibration=cam_calibration)
        img = imread(img_path)
        res = ld.process_frame(img)

    # ld = LaneDetector(SRC, DST, n_frames=FRAME_MEMORY, cam_calibration=cam_calibration)
    #
    # project_output = 'night_out.mp4'
    # clip1 = VideoFileClip('VID_20161210_181736.mp4')
    # project_clip = clip1.fl_image(ld.process_frame)
    # project_clip.write_videofile(project_output, audio=False)