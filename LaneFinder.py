import numpy as np
import cv2
import calib
import preprocess
from glob import glob
import matplotlib.pyplot as plt


class HistogramLaneFinder:
    def __init__(self, vertical_slices):
        # The number of vertical slices in the histogram processing
        self.vertical_slices = vertical_slices

        # The markings for each cycle
        self.markings = []

        # The differences across the histogram
        self.differences = np.zeros(shape=(vertical_slices, 2), dtype=np.float)

    def find(self, img, base_lane_width):
        """
        Finds lane fit equations in the 2nd degree from a bird's eye view of the road
        :param img: Bird's eye view of the road, where the lane lines are parallel
        :param base_lane_width: The lane width at the base
        :return: Left and right fit equations
        """
        # Compute the histogram
        histogram = preprocess.running_mean(img, self.vertical_slices, 50)

        # Search each of the histograms for the lanes
        raw_markings = []
        self.markings = []
        dbg_intervals = []
        for i in np.arange(self.vertical_slices - 1, -1, -1):
            # Get the intervals and lane width from previous values
            intervals, lane_width, lane_width_confidence, left_confidence, right_confidence = \
                self.__get_expected_interval_and_width(img.shape[1])
            dbg_intervals.append(intervals)

            # If we weren't able to find the lane width, take the given base value
            if lane_width is None:
                lane_width = base_lane_width

            # Persistent values across the loop
            prev_pos = 0
            start = 0
            found = []
            features = np.argwhere(histogram[i] > 0)

            for val in features:
                # Find if the value lies within any of the intervals
                if (intervals[0][0] <= val <= intervals[0][1]) or (intervals[1][0] <= val <= intervals[1][1]):
                    if val > prev_pos + 10:
                        # If we've already started to find a interval
                        if start:
                            # Append the results with this interval and start a new interval
                            found.append(np.mean([start, prev_pos]))
                            start = val
                        else:
                            start = val
                    prev_pos = val

                else:
                    # If the value does not lie in any of the expected intervals, then go to the next value,
                    # But before that, in case we 've a lane in the queue, we should add it
                    if start:
                        found.append(np.mean([start, prev_pos]))
                        start = 0

            # Add the final lane marking (if we've a valid lane in the queue)
            if start:
                found.append(np.mean([start, prev_pos]))

            # Find two values that are probable lane candidates from the list of found
            candidates = self.__find_prob_markings(found, lane_width,
                                                   (np.mean(intervals[0]), np.mean(intervals[1])), i,
                                                   lane_width_confidence, left_confidence, right_confidence)

            # Compute the differences
            if i == self.vertical_slices - 1:
                self.differences[i][0] = 0
                self.differences[i][1] = 0
            else:
                if candidates[0] is not None:
                    self.differences[i][0] = candidates[0] - np.mean(intervals[0])
                else:
                    self.differences[i][0] = self.differences[i - 1][0]
                if candidates[1] is not None:
                    self.differences[i][1] = candidates[1] - np.mean(intervals[1])
                else:
                    self.differences[i][1] = self.differences[i - 1][1]

            # Add to the global list
            raw_markings.append(found)
            self.markings.append(candidates)


        return self.markings

    # ------------- STATIC FUNCTIONS ------------- #
    def __get_expected_interval_and_width(self, shape):
        intervals = [[0, shape / 2], [shape / 2, shape]]
        lane_width = None
        lane_width_confidence = 300 - (len(self.markings) * 25)
        left_confidence = 300 - (len(self.markings) * 25)
        right_confidence = 300 - (len(self.markings) * 25)

        # Search from the most recent
        left_found = right_found = False
        for i in np.arange(len(self.markings) - 1, -1, -1):
            # Check if this one is not None
            if not left_found and self.markings[i][0] is not None:
                intervals[0] = [self.markings[i][0] - 200, self.markings[i][0] + 200]
                left_found = True
            if not right_found and self.markings[i][1] is not None:
                intervals[1] = [self.markings[i][1] - 200, self.markings[i][1] + 200]
                right_found = True
            if left_found and right_found:
                break
            if not left_found:
                left_confidence += 50
            if not right_found:
                right_confidence += 50
            lane_width_confidence += 50

        if left_found and right_found:
            lane_width = np.mean(intervals[1]) - np.mean(intervals[0])

        return intervals, lane_width, lane_width_confidence, left_confidence, right_confidence

    def __find_prob_markings(self, curr_finds, lane_width, curr_range, idx,
                             lane_width_confidence, left_confidence, right_confidence):
        """
        Finds probable lane markings from all the marking founds
        :param curr_finds: All the markings found in a single vertical slice histogram
        :param lane_width: The expected lane width at this histogram
        :return: Returns the lane co-ordinates if it was found, or returns None
        """
        answer = [None, None]

        if len(curr_finds):
            if idx != (self.vertical_slices - 1):
                # If this is not the first histogram (bottom row), test against the previous
                # histogram values to find the correct markings
                answer = self.__check_prev_value(curr_finds, curr_range, self.differences[idx + 1],
                                                 left_confidence, right_confidence)

                # Cross verify the values against the lane width if we've both the sides
                if answer[0] is not None and answer[1] is not None:
                    lane_answer = self.__check_width(answer, lane_width, lane_width_confidence)
                    if lane_answer != answer:
                        answer = [None, None]
            else:
                # If this is the bottom row, test against the lane width to find the correct markings
                answer = self.__check_width(curr_finds, lane_width, lane_width_confidence)

        return answer

    # ------------- STATIC FUNCTIONS ------------- #
    @staticmethod
    def __is_in_interval(val, intervals):
        valid = False
        for interval in intervals:
            if val <= interval[1] and val >= interval[0]:
                valid = True
                break

    @staticmethod
    def __check_width(curr_finds, lane_width, lane_width_confidence):
        answer = [None, None]

        results = np.ones(shape=(len(curr_finds), len(curr_finds)), dtype=np.float)
        results *= np.infty

        for i in np.arange(len(curr_finds)):
            if curr_finds[i] is not None:
                for j in np.arange(i + 1, len(curr_finds)):
                    if curr_finds[j] is not None:
                        results[i][j] = np.abs((curr_finds[j] - curr_finds[i]) - lane_width)

        if results.min() < lane_width_confidence:
            # If the lane width between any of the two curr_finds matches the expected lane width,
            # then hurray, we've the answer
            answer = np.unravel_index(results.argmin(), results.shape)
            answer = [curr_finds[answer[0]], curr_finds[answer[1]]]

        return answer

    @staticmethod
    def __check_prev_value(curr_finds, curr_range, differences, left_confidence, right_confidence):
        answer = [None, None]

        results = np.ones(shape=(len(curr_finds), 2), dtype=np.float)
        results *= np.infty

        for i in np.arange(len(curr_finds)):
            results[i][0] = np.abs((curr_finds[i] - curr_range[0]) - differences[0])
            results[i][1] = np.abs((curr_finds[i] - curr_range[1]) - differences[1])

        # Find if any of the markings are valid for one of the sides
        min_args = np.argmin(results, 0)
        minimums = np.min(results, 0)
        if minimums[0] < left_confidence:
            answer[0] = curr_finds[min_args[0]]
        if minimums[1] < right_confidence:
            answer[1] = curr_finds[min_args[1]]
        return answer

class LaneFinder:
    def __init__(self, base_lane_width=700, cam_calibration=None):
        self.constants = {
            'HIST_VERT_SLICES': 10
        }

        # The camera calibration parameters
        self.cam_calibration = cam_calibration

        # The original image's shape
        self.img_sz = None

        # Histogram lane finder
        self.hist_finder = HistogramLaneFinder(self.constants['HIST_VERT_SLICES'])

        # Frame memory related variables
        self.frame_mem_max = 5
        self.frame_memory = None
        self.frame_avail = np.zeros(shape=self.frame_mem_max, dtype=np.bool)
        self.frame_idx = 0

        # Typical lane widths at each of the histogram slices, we initialize the base width alone
        self.lane_width = np.zeros(shape=(self.constants['HIST_VERT_SLICES']), dtype=np.uint32)
        self.lane_width[-1] = base_lane_width

    def process(self, frame):
        # Save the shape of the original image
        self.img_sz = frame.shape

        # Initialize the frame memory with the frame shape
        if self.frame_memory is None:
            self.frame_memory = np.zeros(shape=(self.frame_mem_max, frame.shape[0], frame.shape[1]), dtype = np.uint8)

        # Compute the transformation co-ordinates
        transform_dst = np.array([
            [0, frame.shape[0]],
            [0, 0],
            [frame.shape[1], 0],
            [frame.shape[1], frame.shape[0]]
        ])

        # Find a probable region of interest using canny and hough transform along with the confidence of the ROI
        region_of_interest, left_roi_conf, right_roi_conf = self.__find_roi_using_hough(frame)

        # Mask the image
        marked = np.copy(frame)
        cv2.line(marked,
                 (region_of_interest[0][0], region_of_interest[0][1]),
                 (region_of_interest[1][0], region_of_interest[1][1]),
                 (255, 0, 0), 10)
        cv2.line(marked,
                 (region_of_interest[2][0], region_of_interest[2][1]),
                 (region_of_interest[3][0], region_of_interest[3][1]),
                 (255, 0, 0), 10)

        # Apply the distortion correction to the raw image.
        if self.cam_calibration is not None:
            dist_correct = calib.undistort(frame, self.cam_calibration)
        else:
            dist_correct = np.copy(frame)

        # Transform the perspective
        perspective_transformer = preprocess.PerspectiveTransformer(np.float32(region_of_interest), np.float32(transform_dst))
        birdseye = perspective_transformer.transform(dist_correct)
        birdseye = preprocess.median_blur(birdseye, 25)

        # Use color transforms, gradients, etc., to create a thresholded binary image.
        binary = self.__threshold_image(birdseye)
        binary = preprocess.guassian_blur(binary, 25)

        # Add the image to the frame memory
        self.frame_memory[self.frame_idx] = binary
        self.frame_avail[self.frame_idx] = True
        self.frame_idx = (self.frame_idx + 1) % self.frame_mem_max

        # Compute a motion blur across the frame memory
        motion = np.zeros(shape=(self.img_sz[0], self.img_sz[1]), dtype=np.uint8)
        for idx, img in enumerate(self.frame_memory):
            if self.frame_avail[idx]:
                motion[(img > .5)] = 1

        # Search for the lanes using sliding windows
        lane_markings = self.hist_finder.find(motion, self.lane_width[-1])

        # Draw boxes for each found marking
        box_marked = np.copy(birdseye)
        size = frame.shape[0] / self.constants['HIST_VERT_SLICES']
        for idx, marking in enumerate(lane_markings):
            start = int(size * (self.constants['HIST_VERT_SLICES'] - idx - 1))
            end = int(size * (self.constants['HIST_VERT_SLICES'] - idx))
            if marking[0] is not None:
                cv2.rectangle(box_marked, (int(marking[0]) - 50, end), (int(marking[0]) + 50, start), (255), 10)
            if marking[1] is not None:
                cv2.rectangle(box_marked, (int(marking[1]) - 50, end), (int(marking[1]) + 50, start), (255), 10)

        # Perform perspective transform
        fig = plt.figure()
        fig.add_subplot(331), plt.imshow(frame), plt.title('Original')
        fig.add_subplot(332), plt.imshow(marked), plt.title('ROI marked')
        fig.add_subplot(333), plt.imshow(dist_correct), plt.title('Distortion corrected')
        fig.add_subplot(334), plt.imshow(birdseye), plt.title('Bird\'s view')
        fig.add_subplot(335), plt.imshow(binary, cmap='gray'), plt.title('Thresholded')
        fig.add_subplot(336), plt.imshow(motion, cmap='gray'), plt.title('Motion Blur')
        fig.add_subplot(337), plt.imshow(box_marked, cmap='gray'), plt.title('Box Marked')
        plt.show()

        return marked

    # ------------- PRIVATE FUNCTIONS ------------- #
    def __find_prob_markings(self, markings):
        answer = None

        results = np.ones(shape=(len(markings), len(markings)), dtype=np.float)
        results *= np.infty

        for i in np.arange(len(markings)):
            for j in np.arange(i+1, len(markings)):
                results[i][j] = np.abs((np.mean(markings[j]) - np.mean(markings[i])) - self.lane_width)

        if results.min() < 200:
            answer = np.unravel_index(results.argmin(), results.shape)
            answer = [markings[answer[0]], markings[answer[1]]]

        return answer

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
    def __find_roi_using_hough(img, fig=None):
        try:
            IMG_SZ = img.shape[0:2]
            bnw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            flt = preprocess.guassian_blur(bnw, 21)
            cny = preprocess.canny(flt, 40, 50)
            ROI = np.array([[IMG_SZ[1] * 0.0, IMG_SZ[0] * 1.0],
                            [IMG_SZ[1] * 0.4, IMG_SZ[0] * 0.6],
                            [IMG_SZ[1] * 0.6, IMG_SZ[0] * 0.6],
                            [IMG_SZ[1] * 1.0, IMG_SZ[0] * 1.0]])
            roi = preprocess.region_of_interest(cny, np.int32([ROI]))
            verts, left_conf, right_conf = preprocess.hough_lines(roi, 1, np.pi/48, 50, 1, 60, (ROI[1][1] * 1.1))

            # If we've not found the lanes
            if verts[0][1] == 0 or verts[1][1] == 0 or verts[2][1] == 0 or verts[3][1] == 0:
                raise ValueError

            # If we've found the lanes, spread around in the x axis and make the ROI have some room
            verts[0][0] -= 200
            verts[1][0] -= 200
            verts[2][0] += 200
            verts[3][0] += 200

            if fig is not None:
                fig.add_subplot(232), plt.imshow(flt, cmap='gray')
                fig.add_subplot(233), plt.imshow(cny, cmap='gray')
                fig.add_subplot(234), plt.imshow(roi, cmap='gray')

        except:
            # When an error occurs, let's just return a black ROI with 0 confidence
            verts = np.array([[IMG_SZ[1] * 0.0, IMG_SZ[0] * 0.9],
                            [IMG_SZ[1] * 0.4, IMG_SZ[0] * 0.6],
                            [IMG_SZ[1] * 0.6, IMG_SZ[0] * 0.6],
                            [IMG_SZ[1] * 1.0, IMG_SZ[0] * 0.9]])
            verts = verts.astype(np.int)
            left_conf = 0
            right_conf = 0

        return verts, left_conf, right_conf


if __name__ == "__main__":
    # Calibrate the camera
    calibration = calib.calibrate_camera('camera_cal', (9, 6), (720, 1280, 3))

    # Find the lanes
    from scipy.misc import imread, imsave
    images = glob('test_images/test*')
    for idx, img_path in enumerate(images):
        img = imread(img_path)
        ld = LaneFinder(cam_calibration=calibration)
        res = ld.process(img)
        #imsave('output_images/test'+str(idx + 1)+'.jpg', res)

    # from moviepy.editor import VideoFileClip
    # ld = LaneFinder(cam_calibration=calibration)
    # project_output = 'test.mp4'
    # clip1 = VideoFileClip('harder_challenge_video.mp4')
    # project_clip = clip1.fl_image(ld.process)
    # project_clip.write_videofile(project_output, audio=False)