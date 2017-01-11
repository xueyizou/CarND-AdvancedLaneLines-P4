import numpy as np
import cv2
import calib
import preprocess

from glob import glob
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip




class LaneDetector():
    def __init__(self, perspective_src, perspective_dst, cam_calibration=None):
        self.cam_calibration = cam_calibration

        self.perspective_src = perspective_src
        self.perspective_dst = perspective_dst
        self.perspective_transformer = PerspectiveTransformer(perspective_src, perspective_dst)

        self.org_img_size = None
        self.proc_img_size = (1280, 720)

        self.confidences = np.zeros(shape=2, dtype=np.uint32)
        self.max_confidences = np.zeros(shape=2, dtype=np.uint32)
        self.fit_values = np.zeros(shape=(2,3), dtype=np.float)
        self.fit_values[0][2] = 200
        self.fit_values[1][2] = 1100
        self.lane_width = 1000

        self.frame_mem_max = 5
        self.frame_memory = np.zeros(shape=(self.frame_mem_max, 720, 1280), dtype = np.uint8)
        self.frame_avail = np.zeros(shape=self.frame_mem_max, dtype=np.bool)
        self.frame_idx = 0

    def process_frame(self, frame):
        perspective_destination = np.float32([
        (0, 720),
        (0, 0),
        (1280, 0),
        (1280, 720)])

        # Save the shape of the original image
        self.org_img_size = frame.shape

        # Resize the image to a fixed resolution
        resize = cv2.resize(frame, self.proc_img_size, interpolation=cv2.INTER_AREA)

        # Apply the distortion correction to the raw image.
        if self.cam_calibration is not None:
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

        # Add the image to the frame memory
        self.frame_memory[self.frame_idx] = birdseye
        self.frame_avail[self.frame_idx] = True
        self.frame_idx = (self.frame_idx + 1) % self.frame_mem_max

        # Compute a motion blur across the frame memory
        motion = np.zeros(shape=(self.proc_img_size[1], self.proc_img_size[0]), dtype=np.uint8)
        for idx, img in enumerate(self.frame_memory):
            if self.frame_avail[idx]:
                motion[(img > .5)] = 1

        # Find the lanes
        left_fit, left_confidence, right_fit, right_confidence = self.__find_lanes(motion)

        # Compute the softmax confidences
        curr_confidence_left = self.__compute_confidence(self.confidences[0], left_confidence)
        curr_confidence_right = self.__compute_confidence(self.confidences[1], right_confidence)

        # Compute the averaged fit values
        if left_fit is not None:
            left_fit = (self.fit_values[0] * curr_confidence_left[0]) + (left_fit * curr_confidence_left[1])
        else:
            left_fit = self.fit_values[0]

        if right_fit is not None:
            right_fit = (self.fit_values[1] * curr_confidence_right[0]) + (right_fit * curr_confidence_right[1])
        else:
            right_fit = self.fit_values[1]

        # Save the confidences for the current frame
        self.confidences = [
            (self.confidences[0] * curr_confidence_left[0]) + (left_confidence * curr_confidence_left[1]),
            (self.confidences[1] * curr_confidence_right[0]) + (right_confidence * curr_confidence_right[1])
        ]

        # Compute the maximums
        if self.max_confidences[0] < self.confidences[0]:
            self.max_confidences[0] = self.confidences[0]
        if self.max_confidences[1] < self.confidences[1]:
            self.max_confidences[1] = self.confidences[1]

        # Draw the lanes
        overlay, result = self.__draw_lanes(frame, motion, left_fit, right_fit)

        # Save the new values
        self.fit_values = [left_fit, right_fit]

        # histogram = self.__running_mean(motion, 10, 50)
        # features = np.mean(histogram, 0)
        # fig = plt.figure()
        # fig.add_subplot(331), plt.imshow(frame), plt.title('Original')
        # fig.add_subplot(332), plt.imshow(resize), plt.title('Resized')
        # fig.add_subplot(333), plt.imshow(dist_correct), plt.title('Undistorted')
        # fig.add_subplot(334), plt.imshow(blur), plt.title('Filtered')
        # fig.add_subplot(335), plt.imshow(binary, cmap='gray'), plt.title('Thresholded')
        # fig.add_subplot(336), plt.imshow(birdseye, cmap='gray'), plt.title('Overhead')
        # fig.add_subplot(337), plt.imshow(overlay), plt.title('Overlay')
        # fig.add_subplot(338), plt.plot(features), plt.title('Motion')
        # fig.add_subplot(339), plt.imshow(result), plt.title('Output')
        # plt.show()
        #
        # return result

    # ------------- PRIVATE FUNCTIONS ------------- #
    def __draw_lanes(self, img, birdseye, left_fit, right_fit):
        """
        Draws the lanes specified by the fits on top of the given image
        :param img: The original image (should be of size img_sz
        :param left_fit: Left fit equation
        :param right_fit: Right fit equation
        :return: Final image
        """
        # Generate the x and y for each lane boundary
        right_y = np.arange(11) * self.proc_img_size[0] / 10
        right_x = right_fit[0] * right_y ** 2 + right_fit[1] * right_y + right_fit[2]

        left_y = np.arange(11) * self.proc_img_size[0] / 10
        left_x = left_fit[0] * left_y ** 2 + left_fit[1] * left_y + left_fit[2]

        # Cast the points into a form that's easy for cv2.fillPoly
        temp = np.zeros_like(birdseye).astype(np.uint8)
        overlay = np.dstack((temp, temp, temp))
        left_pts = np.array([np.transpose(np.vstack([left_x, left_y]))])
        right_pts = np.array([np.flipud(np.transpose(np.vstack([right_x, right_y])))])
        pts = np.hstack((left_pts, right_pts))

        # Compute the center of the image to get the deviation
        left_bot = self.__evaluate_curve(self.proc_img_size[0], left_fit)
        right_bot = self.__evaluate_curve(self.proc_img_size[0], right_fit)
        val_center = (left_bot + right_bot) / 2.0
        self.lane_width = right_bot - left_bot

        # Compute the offset from the middle
        dist_offset = val_center - self.proc_img_size[1] / 2
        dist_offset = np.round(dist_offset / 2.81362, 2)
        str_offset = 'Lane deviation: ' + str(dist_offset) + ' cm.'

        # If we are deviating too much from the middle, make it red
        if dist_offset > 150:
            cv2.fillPoly(overlay, np.int_([pts]), (255, 0, 0))
        else:
            cv2.fillPoly(overlay, np.int_([pts]), (0, 255, 0))

        # Draw the lane onto the warped blank image
        self.__draw_line(overlay, np.int_(left_pts), (255, 255, 255))
        self.__draw_line(overlay, np.int_(right_pts), (255, 255, 255))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        overlay_perspective = self.perspective_transformer.inverse_transform(overlay)

        # Resize the overlay back to the original image dimension and combine with the original image
        overlay_perspective = cv2.resize(overlay_perspective,
                                         dsize=(self.org_img_size[1], self.org_img_size[0]), interpolation=cv2.INTER_AREA)
        result = cv2.addWeighted(img, 1, overlay_perspective, 0.5, 0)

        # Compute curvature
        left_curve = self.__compute_curvature(left_fit, self.proc_img_size[0] / 2)
        Right_curve = self.__compute_curvature(right_fit, self.proc_img_size[0] / 2)
        str_curv = 'Curvature: Right = ' + str(np.round(Right_curve, 2)) + ', Left = ' + str(np.round(left_curve, 2))
        str_conf = 'Confidence: Right = ' + str(np.round(self.confidences[1] / np.max(self.max_confidences), 2)) + \
                   ', Left = ' + str(np.round(self.confidences[0] / np.max(self.max_confidences), 2))

        # Write the curvature and offset values onto the image
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(result, str_curv, (30, 60), font, 1, (255, 0, 0), 2)
        cv2.putText(result, str_offset, (30, 90), font, 1, (255, 0, 0), 2)
        cv2.putText(result, str_conf, (30, 120), font, 1, (255, 0, 0), 2)

        return overlay, result

    def __find_lanes(self, birdseye):
        """
        Finds lane fit equations in the 2nd degree from a bird's eye view of the road
        :param birdseye: Bird's eye view of the road, where the lane lines are parallel
        :return: Left and right fit equations
        """
        # Compute the histogram
        histogram = self.__running_mean(birdseye, 10, 50)

        # Find the features in the histogram where the value is greater than the threshold of .05
        features = np.argwhere(np.mean(histogram, 0) > 0)

        # fig = plt.figure()
        # fig.add_subplot(221), plt.imshow(birdseye, cmap='gray')

        # Compute the left fit
        left_fit = None
        left_values = []
        left_features = features[features < birdseye.shape[1] / 2.]
        if len(left_features) >= 10:
            left_min = np.min(left_features)
            left_max = np.max(left_features)

            left_image = np.copy(birdseye)
            left_image[:, 0:left_min] = 0
            left_image[:, left_max:birdseye.shape[1]] = 0
            # left_image = preprocess.guassian_blur(left_image, 25)
            # fig.add_subplot(223), plt.imshow(left_image, cmap='gray')

            left_values = np.argwhere(left_image > .5)
            if len(left_values) >= 10:
                all_x = left_values.T[0]
                all_y = left_values.T[1]
                left_fit = np.polyfit(all_x, all_y, 2)

        # Compute the right fit
        right_fit = None
        right_values = []
        right_features = features[features > birdseye.shape[1] / 2.]
        if len(right_features) >= 10:
            right_min = np.min(right_features)
            right_max = np.max(right_features)

            right_image = np.copy(birdseye)
            right_image[:, 0:right_min] = 0
            right_image[:, right_max:birdseye.shape[1]] = 0
            # right_image = preprocess.guassian_blur(right_image, 25)
            # fig.add_subplot(224), plt.imshow(right_image, cmap='gray')

            right_values = np.argwhere(right_image > .5)
            if len(right_values) >= 10:
                all_x = right_values.T[0]
                all_y = right_values.T[1]
                right_fit = np.polyfit(all_x, all_y, 2)

        # plt.show()

        # Check if one of the lanes is not confident or does not exist
        # if (left_fit is not None) and (len(left_values) < len(right_values)) and right_fit is not None:
        #     ratio = .5 + (.5 * len(left_values) / len(right_values))
        #     left_fit[0] = (left_fit[0] * ratio) + (right_fit[0] * (1 - ratio))
        #     left_fit[1] = (left_fit[1] * ratio) + (right_fit[1] * (1 - ratio))
        #     left_fit[2] = (left_fit[2] * ratio) + ((right_fit[2] - self.lane_width) * (1 - ratio))
        # if (right_fit is not None) and (len(right_values) < len(left_values)) and left_fit is not None:
        #     ratio = .5 + (.5 * len(right_values) / len(left_values))
        #     right_fit[0] = (right_fit[0] * ratio) + (left_fit[0] * (1 - ratio))
        #     right_fit[1] = (right_fit[1] * ratio) + (left_fit[1] * (1 - ratio))
        #     right_fit[2] = (right_fit[2] * ratio) + ((left_fit[2] + self.lane_width) * (1 - ratio))

        return left_fit, len(left_values), right_fit, len(right_values)

    # ------------- STATIC FUNCTIONS ------------- #
    @staticmethod
    def __threshold_image(img, draw=False):
        """
        Thresholds an image based on various criteria
        :param img: Image to be thresholded
        :param draw: If the results should be plotted
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

        if draw:
            fig = plt.figure()
            fig.add_subplot(331), plt.imshow(img)
            fig.add_subplot(332), plt.imshow(gamma, cmap='gray'), plt.title('Gamma')
            fig.add_subplot(333), plt.imshow(white, cmap='gray'), plt.title('White')
            fig.add_subplot(334), plt.imshow(yellow, cmap='gray'), plt.title('Yellow')
            fig.add_subplot(335), plt.imshow(color, cmap='gray'), plt.title('Color')
            fig.add_subplot(336), plt.imshow(sobel_l, cmap='gray'), plt.title('Sobel L')
            fig.add_subplot(337), plt.imshow(sobel_s, cmap='gray'), plt.title('Sobel S')
            fig.add_subplot(338), plt.imshow(sobel, cmap='gray'), plt.title('Sobel')
            fig.add_subplot(339), plt.imshow(result, cmap='gray'), plt.title('Final')
            plt.show()

        return result

    @staticmethod
    def __draw_line(img, pts, color):
        """
        Draws a line given a series of points
        :param img: The image on which line should be drawn
        :param pts: Points to draw the image
        :param color: Color of the line
        :return: None
        """
        pts = np.int_(pts)
        for i in np.arange(len(pts[0]) - 1):
            x1 = pts[0][i][0]
            y1 = pts[0][i][1]
            x2 = pts[0][i + 1][0]
            y2 = pts[0][i + 1][1]
            cv2.line(img, (x1, y1), (x2, y2), color, 50)

    @staticmethod
    def __compute_confidence(prev_confidence, frame_confidence):
        """Compute confidence values for each sets of score
        :param frame_confidence: The confidence of the current frame
        :param prev_confidence: The confidence of all the prediction till now
        """
        if prev_confidence:
            val = (frame_confidence / prev_confidence) * .1
            return [1. - val, val]
        else:
            return [0., 1.]

    @staticmethod
    def __evaluate_curve(y, fit):
        """
        Evlauates a curve fit at some y
        :param y: The y at which the curve fit should be evaluated
        :param fit: The curve parameters
        :return: The value of x at the position y
        """
        return fit[0] * (y ** 2) + fit[1] * y + fit[2]

    @staticmethod
    def __compute_curvature(pol_a, y_pt):
        """
        Computes the curvature given a line fit
        """
        A = pol_a[0]
        B = pol_a[1]
        result = (1 + (2 * A * y_pt + B) ** 2) ** 1.5 / 2 / A
        return result

    @staticmethod
    def __running_mean(img, vert_slices, wsize):
        """
        Computes the horizontal moving histogram of an image
        :param img: The binary image (ch = 2)
        :param vert_slices: Number of vertical slices
        :param wsize: The window size
        :return: The computed histograms
        """
        size = img.shape[0] / vert_slices
        result = np.zeros(shape=(vert_slices, img.shape[1]), dtype=np.float)

        for i in np.arange(vert_slices):
            start = i * size
            end = (i + 1) * size
            vertical_mean = np.mean(img[start:end], axis=0)

            for j in np.arange(wsize / 2):
                vertical_mean = np.insert(vertical_mean, 0, vertical_mean[0])
                vertical_mean = np.insert(vertical_mean, len(vertical_mean), vertical_mean[-1])

            window_sum = np.cumsum(vertical_mean)
            result[i, :] = (window_sum[wsize:] - window_sum[:-wsize]) / wsize

        return result
if __name__ == "__main__":
    SRC = np.float32([
        (0, 720),
        (530, 470),
        (750, 470),
        (1280, 720)])

    DST = np.float32([
        (0, 720),
        (0, 0),
        (1280, 0),
        (1280, 720)])

    calibration = calib.calibrate_camera('camera_cal', (9, 6), (720, 1280, 3))

    # from scipy.misc import imread, imsave
    # images = glob('test_images/test*')
    # for idx, img_path in enumerate(images):
    #     img = imread(img_path)
    #     ld = LaneDetector(SRC, DST, cam_calibration=calibration)
    #     res = ld.process_frame(img)
    #     imsave('output_images/test'+str(idx + 1)+'.jpg', res)


    ld = LaneDetector(SRC, DST, cam_calibration=calibration)
    project_output = 'project_video_out.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    project_clip = clip1.fl_image(ld.process_frame)
    project_clip.write_videofile(project_output, audio=False)