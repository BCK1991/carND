# **Finding Lane Lines on the Road** 


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. 

1- Convert input image to grayscale via grayscale()

![1](https://user-images.githubusercontent.com/30300329/28388896-78c2f9b4-6cd4-11e7-8a90-f4dea83f2169.png)

2- Apply Gaussian smoothing to reduce high frequency items and noise

3- Detect edges via Canny transform

![2](https://user-images.githubusercontent.com/30300329/28388904-7d359394-6cd4-11e7-98e4-68bc8969d29b.png)

4- Define a four-sided polygon and mask the image
5- Define Hough transform coefficients, detect and draw lines by calling HoughLineP()

![3](https://user-images.githubusercontent.com/30300329/28388913-814acc92-6cd4-11e7-83fc-8d0fab600ba7.png)

6- Merge initial image with the resulted lines 

![4](https://user-images.githubusercontent.com/30300329/28388916-835e16ba-6cd4-11e7-9aa6-2fa3838f488c.png)

In order to draw a single line on the left and right lanes, I modified the draw_lines() function to first allocate the lines as left and right lines based on the slope and the the x-coordinates. Moreover, I fine tuned the slope selection to segregate noisy lines.
Then, I fit the points to a line seperately for left and right. Based on the slope (m), offset (b) and known coordinates of the y-values, I calculate x-values. Finally, cv2.line function is called to draw lines for calculated x-y values seperately for the left and the right clusters.


### 2. Identify potential shortcomings with your current pipeline


I think in general the area of interest might differ depending on the camera, position of the camera in the car and the position of the car in the lane. There might be even more situation that current polygon cannot handle. After seeing the challenge part and start debugging, I can already tell my pipeline is not seeing the right lane lines at all.

I think any defect on the road (in the lane of the car) might create an issue for the pipeline and can result as a false positive.

The hough transform coefficients (especially the threshold, min_line_lenght and max_line_gap) are difficult to adjust to have a one-fits-all configuration. 

A more dynamic approach would be required for all the abovementioned points. Instead of looking for a one-fits-all configuration, runtime corrections and adaptation should be enabled by extra set of algorithms to cover different environment settings.


### 3. Suggest possible improvements to your pipeline

In general, finding a one-fits-all configuration by using my current toolbox (algorithms that we learnt so far) by tweeking a set of parameters limits the flexibility and scalability. An advance approach that utilizes more tools seem to be required.

On the other hand, I could've spent more time on tweeking to fine-tune the final lines.
