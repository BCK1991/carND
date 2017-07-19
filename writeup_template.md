# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

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
2- Apply Gaussian smoothing to reduce high frequency items and noise
3- Detect edges via Canny transform
4- Define a four-sided polygon and mask the image
5- Define Hough transform coefficients, detect and draw lines by calling HoughLineP()
6- Merge initial image with the resulted lines 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function to first allocate the lines as left and right lines based on the slope and the the x-coordinates. Moreover, I fine tuned the slope selection to segregate noisy lines.
Then, I fit the points to a line seperately for left and right. Based on the slope (m), offset (b) and known coordinates of the y-values, I calculate x-values. Finally, cv2.line function is called to draw lines for calculated x-y values seperately for the left and the right clusters.

If you'd like to include images to show how the pipeline works, here is how to include an image: 

[image2]: /steps_images/1.png "After step 1"


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
