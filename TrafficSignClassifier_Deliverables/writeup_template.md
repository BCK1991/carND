#**Traffic Sign Recognition** 




---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Dataset.JPG "Distribution over the different signs"
[image2]: ./Preprocessing.JPG "Before and After Preprocessing"
[image3]: ./CNN_Architecture.JPG "CNN Architecture"
[image4]: ./sign1.jpg "Traffic Sign 1"
[image5]: ./sign2.jpg "Traffic Sign 2"
[image6]: ./sign3.jpg "Traffic Sign 3"
[image7]: ./sign4.jpg "Traffic Sign 4"
[image8]: ./sign5.jpg "Traffic Sign 5"
[image9]: ./sign6.jpg "Traffic Sign 6"
[image10]: ./sign7.jpg "Traffic Sign 7"
[image11]: ./sign8.jpg "Traffic Sign 8"
[image12]: ./sign9.jpg "Traffic Sign 9"
[image13]: ./Type3.JPG "Type3"
[image14]: ./Type11.JPG "type11"
[image15]: ./Tpye29.JPG "type29"
[image16]: ./Type33.JPG "type33"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/BCK1991/carND/blob/master/TrafficSignClassifier_Deliverables/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Info about the dataset

I used the pandas library to calculate summary statistics of the traffic
signs data set:


* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 42

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the dataset is composed, i.e. number of images per label. It shows that dataset is not really uniformly distributed over different signs. One possible workaround is to augmenting the data set by using simple rotations for images that shows symmetry. 

![alt text][image1]

Here are some more examples from the dataset. As can be seen, some of the images are even difficult for human to recognize due reflection, contrast and even the angle of shot. 

![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. 

As a first step, I decided to convert the images to grayscale to decrease the memory and computational overhead. On top, I've recently read a paper, "Using Grayscale Images for Object Recognition with Convolutional-Recursive Neural Network", **Bui et. al, 2016, IEEE Conference Publications, p.321 - 325**, that concludes classification with grayscale images results in higher accuracy classification compared with RGB.

As a second step, I normalized (-1,1) the image data to bring the ill-conditioned data to well-conditioned (centered) status to ensure the negative gradient to show directly to the center (minima).

Here is an example of a traffic sign image before and after pre-processing(grayscale + normalization).

![alt text][image2]



I decided NOT to generate additional data due to time limitation, however, I am planning to do it when I have some time available.


####2. Final model architecture

My final model looks like the following:
![alt text][image3]

 
My model uses LeNeT lab as starting point, and I added another fully-connected layer. It looks as following:

- Input (32x32x1)
- Conv Layer (9x9x32 filter, stride 1 to 24x24x32) (padding 'VALID')
- Activation (Sigmoid)
- Max Pooling (2x2 kernel, 2x2 stride to 12x12x32)
- Conv Layer (4x4x84 filter, stride 1 to 9x9x84) (padding 'VALID')
- Activation (Sigmoid)
- Max Pooling (3x3 kernel, 3x3 stride to 3x3x84)
- Flatten (756 nodes)
- Fully Connected (378)
- Fully Connected (84)
- Output (42)


####3.Model Training

I've taken my solution for the LeNeT lab as a starting point and tried to optimize on it, so the initial starting points are coming from my LeNeT lab which has proven itself.
To train the model, I used AdamOptimizer which is similar to GD optimizer but uses Adam algorithm that simply helps algorithm to converge faster compared with GDO.
I decided to keep batch size of 128 to prevent memory issues.
I decided on 30 Epochs after some trials. Throughout the EPOCHS, I saved my parameters if the accuracy of the last iteration is bigger than the previously saved parameters. In this way, if the flactuations in the accuracy causes drop in the last iterations, I still keep the parameters from the iteration that yields the maximum accuracy.
Learning rate is set to 0.0008 to lower the accuracy flactuations.
I also decided to use sigmoid as activation function to minimze the number of dead neurons. I've also observed, with ReLu, the accuracy suddenly dies after 5-6 EPOCHS (drops from 0.80-0.900 to 0.050-0.060) which was not the case with Sigmoid.

- EPOCHS = 30
- Learning rate = 0.0008
- BATCH_SIZE = 128

####4. Approach

As mentioned above, I've used the architecture from the LeNeT lab to start with. It has 2 convolutional layers + 2 fully connected layers.

1- I've seen a significant drop in the accuracy during the training (after 5-6 EPOCHS drops from 0.80-0.900 to 0.050-0.060) I suspected from underfitting.

2- I decided to increase the number of filter layers to increase the efficiency and prevent underfitting. (More filter layers = more features to be detected)

3- Since I also decided to use maxpool with 2x2, 2x2 and 3x3, 3x3 that decreased the number of pixels, I added more filter layers.

4- As I did not want to explode the memory and/or computational effort, I start increasing the number of filter depth with small stepsize, hence the size of the entire network.


5- After a number of iterations, accuracy reached the expected levels, however, I observed some flactutation that was preventing a reliable accuracy above 0.93. I decided to save the parameters for each iteration that yields a better accuracy.

5- Maybe one important design change was to use Sigmoid activation function instead of ReLu to minimize dead neurons. (mentioned above).

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.933 
* test set accuracy of 0.898



###Test a Model on New Images

####1. Choose nine German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are nine German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 

The first image might be difficult to classify because there are 9 other similar images (speed limits) in the same dataset.
The second image might be difficult to classify as it is underrepresented in the training/test/validation datasets.
The third image has poorer quality (handpicked) and a bit of capture angle (picture is not taken with a 90 degrees angle). And it is also underrepresented in the T/T/V datasets.

![alt text][image7] ![alt text][image8] ![alt text][image9]

The fourth image is also a maximum speed limit sign and might be difficult for the network to differentiate from the other speed limit signs.
The fifth one should be relatively easy as it is unique in terms of the shape and colors. Also it is well-represented in the datasets.
The sixth and eight one might be difficult to classify due to their undistinctive structure (triangular shape, black sign in the middle).

![alt text][image10] ![alt text][image11] ![alt text][image12]

The seventh one also should be relatively easy as it has a simple shape without details and it is well-represented in the datasets.
The ninth one might be difficult to classify due to low resolution as the middle three stripes are not distinguishable.



####2. Model Prediction

Let's remember the all the results:

* training set accuracy 0.994
* validation set accuracy of 0.933 
* test set accuracy of 0.898
* new images set accuracy of 0.889

As the training set accuracy is higher than the rest, the model is overfitting. The straight-forward way would be to use regularization techniques, specifically drop-out, as it is mostly used to remedy overfitting, so I'd first try dropout with very low percentages and keep increasing. I think I could also do data augmentation, but as I mentioned above, I was already a bit delayed, so I did not want to spend time on it. 

I usually start with fine tuning the hyper-parameters, and I think in my case it worked (at least the difference in the accuracy is not too big). ( I will still do data augmentation and dropout as well as visualization when I have some free time, I think these are the real selling points :) )

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 km/h      		| 60 km/h   									| 
| Keep Left     			| Keep Left 										|
| Turn Left Ahead					| Turn Left Ahead							|
| 20 km/h	      		| 20 km/h						 				|
| Priority Road			| Priority Road      							|
| General Caution			| General Caution    							|
| Yield		  | Yield      							|
| Double Curve		  | Children Crossing    							|
| End of speed limits		  | End of speed limits		|


The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 89%. This compares favorably to the accuracy on the test set of 0.898.

####3. Model performance on the new images and softmax probabilites

The code for making predictions on my final model is located in the 20th and 21st cells of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .52         			| 60 km/h   									| 
| .25     				| 30 km/h 										|
| .06					| End of speed limit 80					|
| .03	      			| Vehicles over 3.5 metric tons..		|
| .02				    | 100 km/h   							|



For the second image the model is almost certain (99.6%) that this is a 'Keep Left' sign, and it is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Keep Left   									| 
| .00     				|  Go straight or left							|
| .00					| X											|
| .00	      			| X					 				|
| .00				    | X    							|

For the third image the model is almost certain (99.4%) that this is a 'Turn Left Ahead' sign, and it is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Keep Left   									| 
| .00     				|  Go straight or right							|
| .00					| X											|
| .00	      			| X					 				|
| .00				    | X    							|

For the fourth image the model is almost certain (99.9%) that this is a '30 km/h (speed limit)' sign, and it is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 30 km/h   									| 
| .00     				|  20 km/h							|
| .00					| X											|
| .00	      			| X					 				|
| .00				    | X    							|

For the fifth image the model is almost certain (99.8%) that this is a 'Priority road' sign, and it is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Priority road									| 
| .00     				|  Roundabout man.							|
| .00					| X											|
| .00	      			| X					 				|
| .00				    | X    							|

For the sixth image the model is almost certain (99.8%) that this is a 'General Caution' sign, and it is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| General Caution									| 
| .00     				|  Ahead only							|
| .00					| X											|
| .00	      			| X					 				|
| .00				    | X    							|

For the seventh image the model is almost certain (99.8%) that this is a 'Yield' sign, and it is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Yield									| 
| .00     				|  Ahead only							|
| .00					| X											|
| .00	      			| X					 				|
| .00				    | X    							|

For the eighth image the model is half confident (99.8%) that this is a 'Children crossing' sign, and but it is a 'Double curve' sign. Double curve sign is not even in the top 5 of the probabilities. This might be due to under-representation of the sign and/or similarities with other signs (Dangerous curve to the right etc.), which is in the list

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .50         			| Children crossing									| 
| .35     				|  Dangerous curve to the right							|
| .03					| Slippery Road											|
| .02	      			| Bicycle crossing					 				|
| .02				    | 20 km/h   							|

For the nineth image the model is also almost certain (97%) that this is a 'End of all speed and passing limits' sign, and it is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| End of all speed and passing limits									| 
| .00     				|  End of no passing							|
| .00					| X											|
| .00	      			| X					 				|
| .00				    | X    							|




