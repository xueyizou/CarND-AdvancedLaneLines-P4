import numpy as np
import cv2


class Sobel:
    @staticmethod
    def absolute_thresh(img, orientation='x', sobel_kernel=3, threshold=(0, 255)):
        """
        Computes the absolute threshold of an image, either in the x or in the y direction
        :param img: The image to be processed
        :param orientation: Orientation of the sobel processing ('x' or 'y'), by default it's 'x'
        :param sobel_kernel: The size of the sobel kernel. Larger the size, the result is more smoothened. By default
        it's 3.
        :param threshold: The threshold for the resulting image, by default, it's the entire threshold value
        :return: Image after threshold computation
        """
        if orientation == 'x':
            axis = (1, 0)
        elif orientation == 'y':
            axis = (0, 1)
        else:
            raise ValueError('orientation has to be "x" or "y" not "%s"' % orientation)

        sobel = cv2.Sobel(img, cv2.CV_64F, *axis, ksize=sobel_kernel)
        sobel = np.absolute(sobel)

        scale_factor = np.max(sobel) / 255
        sobel = (sobel / scale_factor).astype(np.uint8)

        result = np.zeros_like(sobel)
        result[(sobel > threshold[0]) & (sobel <= threshold[1])] = 1

        return result

    @staticmethod
    def magnitude_thresh(img, sobel_kernel=3, threshold=(0, 255)):
        """
        Computes the magnitude threshold of an image
        :param img: The image to be processed
        :param sobel_kernel: The size of the sobel kernel. Larger the size, the result is more smoothened. By default
        it's 3.
        :param threshold: The magnitude threshold for the resulting image
        :return: Image after threshold computation
        """
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        scale_factor = np.max(magnitude) / 255
        magnitude = (magnitude / scale_factor).astype(np.uint8)

        result = np.zeros_like(magnitude)
        result[(magnitude > threshold[0]) & (magnitude <= threshold[1])] = 1

        return result

    @staticmethod
    def direction_threshold(img, sobel_kernel=3, threshold=(0, np.pi / 2)):
        """
        Computes the direction threshold of an image
        :param img: The image to be processed
        :param sobel_kernel: The size of the sobel kernel. Larger the size, the result is more smoothened. By default
        it's 3.
        :param threshold: The direction threshold for the resulting image
        :return: Image after threshold computation
        """
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        with np.errstate(divide='ignore', invalid='ignore'):
            direction = np.absolute(np.arctan(sobel_y / sobel_x))
            result = np.zeros_like(direction)
            result[(direction > threshold[0]) & (direction <= threshold[1])] = 1

        return result


def gamma_threshold(img, gamma=1.0, threshold=(100, 255)):
    """
    build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    :param img: Image to be processed
    :param gamma: Gamma correction value for the original image (in range of 0.0 to 1.0)
    :param threshold: Treshold to apply for gamma correction
    :return: Returns the gamma thresholded image
    """
    #
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    result = cv2.LUT(img, table)

    # Threshold the image
    result = np.mean(result, 2)
    _, result = cv2.threshold(result.astype(np.uint8), threshold[0], threshold[1], cv2.THRESH_BINARY)

    return result
