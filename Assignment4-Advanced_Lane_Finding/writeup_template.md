## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./P1_Undist_Chess.jpg "Undisturbed Chess Board"
[image2]: ./P2_InputImg.jpg "Test Image"
[image3]: ./P3_UndistImg.jpg "Test Image - After Undistortion"
[image4]: ./P4_BinaryImg.jpg "Binary Example - 1"
[image5]: ./P5_AfterHLS.jpg "Binary Example - 1"
[image6]: ./P6_Persp_Transform.jpg "Before and After the Transformation"
[image7]: ./P7_Lanes_Found.jpg "Binary with Fitted Curves"
[image8]: ./P8_Filled_Lanes.jpg "Image with Filled Lanes"

[video1]: ./project_video_result.mp4 "Final Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the third code cell of the IPython notebook located in "./AdvancedLaneFinding.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. (In the code cell 5) I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

Similar to explanation above, I used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. (In the code cell 5) I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image3]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color, gradient and direction thresholds to generate a binary image (thresholding steps at code cell 6 and 7).  Here's an example of my output for this step. 

![alt text][image4]

Then I've converted BGR image to HLS and added another threshold based on H channel (code cell 9). Here's what it looks like.

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transform_pers()`, which appears in code cell 10 of the notebook.   The `transform_pers()` function takes as inputs an undistored image called (`undistortedImg`). Source (`src`) and destination (`dst`) points are hardcoded based on manual calculations.  I chose the hardcode the source and destination points in the following manner:

``` python
    corners=[[595,450],
            [290,670], 
            [1030,670], 
            [690, 450]]
    src = np.float32([corners[0], corners[1], corners[2], corners[3]])

    offset = 300
    img_size = (colGradBin.shape[1], colGradBin.shape[0])
    dst = np.float32(   [ [offset, 0],
                        [offset, img_size[1]],
                        [img_size[0]-offset, img_size[1]],
                        [img_size[0]-offset, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595, 450      | 300, 0        | 
| 290, 670      | 300, 720      |
| 1030, 670     | 980, 720      |
| 690, 450      | 980, 0        |

Here is an example of input image and output top-down view.


![alt text][image6]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Initial lane finding algorithms utilizes histogram of binary image. It divides the image into 9 pieces (horizontally) and searches for the peaks at left and right (divided into two vertically). It repeats the process for each of the 9 horizontal slices and collects all pixels with peaks. Finally, the peaks are fit into second order polynomials. (code cell 12, find_lane(final_binary))

After using this function to find the lanes for the first pixel, I use 'find_lane2' function to limit the search space on the image based on the position of the previously detected lanes. (code cell 21)

Here is an example of binary with fitted curves.


![alt text][image7]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The function 'find_curvature' calculates the curvature based on lane points. (code cell 20)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code cell 15, 'draw_lane' that takes inverse transform matrix, final threshold binary, undistorted image and polynomial coefficients as input.  It recasts the x and y points into usable format for cv2.fillPoly() and draws the lanes onto warped image. Finally, reverse perspective transfomation takes place.

Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Since I had limited time, I could not really work on mitigating failure conditions for lane finding. A straightforward implementation could just store the poly coeff. from the previous frame and use it in case lane finding fails to detect the lanes. 

I had a chance to try challenge videos and I observed that pipeline is more likely to fail when the light conditions on the road has changed (going through shadows) or when the color of the road has changed. Maybe fine tuning some treshold values would help to overcome the issue but a more advance approach would be to calibrate the threshold value on the air.

In the challenge videos I've also observed my area of interest was a bit of, so in real life scenario the corner coordinates of the area of interest should be calibrated after the camera is mounted on the car.

Finally, also in the challenge video I've observed some detections that are way off-target. A possible solution would be to keep track of (10) previously calculated poly coefficients and compare the newly calculated coefficients with the avarage of previous ones. By simply setting a threshold we could identify if the new coefficients are valid or not and we could disregard the new ones and re-use the values from the previous frame.
