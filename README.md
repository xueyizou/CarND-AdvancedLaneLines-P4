# Advanced Lane Finding
## Introduction
This is Project 4 of the Self Driving Car Nanodegree offerred by 
Udacity. In this project, we need to extend the work that we did 
in [Project1](https://github.com/mohankarthik/CarND-LaneLines-P1),
and find curved lane lines, while also detecting the radius of
 curvature and the deviation from the lane. 

## Software Architecture
The software is organized into the following classes
![Software Architecture](doc_imgs/software_architecture.png)

**Classes**
* preprocess.py: Contains utility functions such as Sobel, 
Canny, Perspective transform, Color Thresholding, Blurring, Hough, etc...
* calib.py: Responsible for Calibrating the camera and removing
camera distortion
* LaneFinder.py: Responsible for finding the lanes given a series of 
image frames
  * This contains two classes, the LaneFinder class and the 
  HistogramLaneFinder class

## Camera Calibration
The first step towards the advanced lane finding is to calibrate
the camera so that there is no distortion in the image that we are
processing. This is important because
> Determining radius of curvature needs the image to be undistored

To do this, we do the following
1. Take a set of images of a standard pattern (like a checkerboard),
using the camera in question. The images must be taken with the
checkerboard at various angles and distances.
2. Find the chessboard corners using `cv2.findChessboardCorners`
3. Use the found points, along with an idealized set of points
to find the distortion / calibration matrix using `cv2.calibrateCamera`
4. Use the calibration matrix to undistort the image using
`cv2.undistort`

After calculating the distortion matrix, I undisorted a test 
image to this effect
![Camera Calibration](doc_imgs/calibration.png)

Next step is to undistort the actual test images and this was 
the results
![Distortion Correction](doc_imgs/distortion.png)
You can notice small changes in the corners of the images that 
has been corrected owing to the distortion correction.

Refer to calib.py for the code that implements Camera Calibration

## First time / Image pipeline
### First time ROI
Region of Interest is critical to most Computer Vision algorithms.
The reason being that searching through the entire image is both
computationally expensive, and difficult in avoiding spurious
image artifacts. So searching within a defined ROI is both 
computationally better, and algorithmically easier.

The first time we see a video, or a single image, the software
has to start searching from scratch. To do this, we use the 
hough transform that we used in the previous project. The 
code is as follows

1. Convert to grayscale `bnw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`
2. Apply a guassian blur `flt = guassian_blur(bnw, 21)`
3. Do Canny edge detection`cny = preprocess.canny(flt, 40, 50)`
4. Apply a wide ROI`roi = preprocess.region_of_interest(cny, np.int32([ROI]))`
5. Find the hough lane lines `verts = preprocess.hough_lines(roi, 1, np.pi/48, 50, 1, 60, ROI[1][1])`

As you can see, this also needs an ROI, which is set to a broad
range which should cover most lane lines in any image.

Running this on the test images gets us the following
![ROI First Time](doc_imgs/roi_first.png)
The hough transform approximately gets the ROI and we add a big
error offset for it, so that we don't inadvertently miss the lane
lines.

The code can be seen in the function __find_roi in LaneFinder.py

### Perspective transform
Now that we've the ROI, we transform this ROI and stretch it to
get our psuedo vertical lane lines. But since we've allowed for
plenty of grace in our ROI find, the perspective transform
will not be vertical for the first few images of our video. Our
dynamic ROI tracker will take care of it later. But at the 
moment, for single images, the transform is going to be only 
partially complete

What we do is the following
1. Use the ROI points found in the previous section as source points
2. Use the image edges as the destination points
3. Compute the transform using `cv2.getPerspectiveTransform`
4. Apply the transform using `cv2.warpPerspective`

Running this on the test images using the ROI got from the previous
section gives us the following
![Perspective Transform](doc_imgs/birdseye.png)
As you can see, not all the transforms gives us perfectly 
parallel lanes. But that's ok. This is a good start.

The code for this can be seen in preprocess.py

### Binary Thresholding

