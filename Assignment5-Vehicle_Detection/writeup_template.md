**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./mk_images/P1-Car_NonCar.JPG "Car and NonCar Examples"
[image2]: ./mk_images/P2-HogFeaturesVis.JPG "HOG Features from RGB ch0"
[image3]: ./mk_images/P1-Car_NonCar.JPG "Car and NonCar Examples"
[image30]: ./mk_images/P3-HogFeaturesHLS-ch0.JPG "HOG Features from HLS ch0"
[image31]: ./mk_images/P3-HogFeaturesHLS-ch1.JPG "HOG Features from HLS ch0"
[image32]: ./mk_images/P3-HogFeaturesHLS-ch2.JPG "HOG Features from HLS ch0"
[image40]: ./mk_images/P5-FindCars1.JPG
[image41]: ./mk_images/P5-FindCars2.JPG
[image42]: ./mk_images/P5-FindCars3.JPG
[image50]: ./mk_images/p7-FrameSeries0.JPG
[image51]: ./mk_images/p7-FrameSeries1.JPG
[image52]: ./mk_images/p7-FrameSeries2.JPG
[image53]: ./mk_images/p7-FrameSeries3.JPG
[image54]: ./mk_images/p7-FrameSeries4.JPG
[image55]: ./mk_images/p7-FrameSeries5.JPG



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/BCK1991/carND/edit/master/Assignment5-Vehicle_Detection/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. HOG Features

The code for this step is contained in lines 116-137 'get_hog_features' of the file 'LessonFunctions.py' 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces in my playground and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space using only channel0[R] and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]


Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]
![alt text][image30]
![alt text][image31]
![alt text][image32]


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I tried to picked RGB to start with since from the visualization I found that one quite promising. However, after I finalize the pipeline, I tried more color spaces (with all 3 channels) and observed that HLS gives the best performance for the project video in hand. Eventually, the HOG parameters I used can be listed as follows:

*cell_per_block = 2
*pixels_per_cel = 8
*orient = 9
*hog_channel = 'ALL'
*colorspace = 'HLS'

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a feature vector composed of spatial, color histogram and hog features. I used a data set of images composed of 5966 car and 5068 non-car images that are 64x64 color images. I divided the data set into two (0.8 for training, 0.2 for testing).

Here are some statistics from the classifier:

*Using spatial binning of: 32 and histogram bins of: 32
Feature vector length: 8460
24.4 Seconds to train SVC...
Test Accuracy of SVC =  0.9973
My SVC predicts:  [ 0.  0.  0.  1.  0.  1.  0.  1.  0.  1.]
For these 10 labels:  [ 0.  0.  0.  1.  0.  1.  0.  1.  0.  1.]
0.015 Seconds to predict 10 labels with SVC*

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search different sizes of windows at different positions with different scaling factors over the image. Here is a table that summarizes the parameters for 'find_cars' function that is at the lines 219-312. Sliding window loops can be find at lines 263-301. The window size is fixed as 64x64 and cells per step (sliding rate) is 4 cells (since cells per block is 8, it corresponds to 50%).

| | range0 | range1 | range2 | range3 |
|-|--------|--------|--------|-------|
| y_start | 450 | 400 | 400 | 380 |
| y_stop | 642 | 592 | 560 | 508 |
| scale | 2.5 | 2 | 1.5 | 1 |



####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image40]


![alt text][image41]


![alt text][image42]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on them:

### Here are six frames with resulting bounding boxes (combined), labels and their corresponding heatmaps:

![alt text][image50]
![alt text][image51]
![alt text][image52]
![alt text][image53]
![alt text][image54]
![alt text][image55]








---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think it was a really fun and challenging project which makes it a good final project for the term-1.

Personally, the most challenging part of the project was to find proper features to be used and good parameters that'd give good results. I spent good amount of time to play around with the parameters and color spaces. Also, since the number of functions used throughout the project was significanlty higher then the previous projects, I needed to follow a bit more systematic way to not make a spagetthi out of the project. I hope it is still readable.

I made use of forum many times throughout the project, unlike the previous ones and I realized it is a strong community which is maybe the most valuable tool for solving the technical problem.

I think the result could be improved especially in terms of the tracking boxes. I realized it was not an easy task to put threshold that both prevents false positives and still captures vehicles without hick-ups. I am planning to spend some more time on the parameters and try to get better performance. Another way would be to add more hog features by using different color spaces and fine tune the sliding window algorithm which I thought would be too much for the purpose of the project. 

