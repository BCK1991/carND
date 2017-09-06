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

[image1]: ./Dataset.jpg "Distribution over the different signs"
[image2]: ./Preprocessing.jpg "Before and After Preprocessing"
[image3]: ./CNN_Architecture.jpg "CNN Architecture"
[image4]: ./sign1.jpg "Traffic Sign 1"
[image5]: ./sign2.jpg "Traffic Sign 2"
[image6]: ./sign3.jpg "Traffic Sign 3"
[image7]: ./sign4.jpg "Traffic Sign 4"
[image8]: ./sign5.jpg "Traffic Sign 5"
[image9]: ./sign6.jpg "Traffic Sign 6"
[image10]: ./sign7.jpg "Traffic Sign 7"
[image11]: ./sign8.jpg "Traffic Sign 8"
[image12]: ./sign9.jpg "Traffic Sign 9"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](carND/TrafficSignClassifier_Deliverables/Traffic_Sign_Classifier.ipynb)

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

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale as color is not a decisive factor while classifying the road signs, however, I still wanted to partially preserve the color information, so I used [0.299,0.587,0.114] coefficients for RGB.

As a second step, I normalized the image data to bring the ill-conditioned data to well-conditioned (centered) status to ensure the negative gradient to show directly to the center (minima).

Here is an example of a traffic sign image before and after pre-processing(grayscale + normalization).

![alt text][image2]



I decided NOT to generate additional data due to time limitation, however, I am planning to do it when I have some time available.


####2. Final model architecture

My final model looks like the following:


 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I've taken my solution for the LeNeT lab as a starting point and tried to optimize on it, so the initial starting points are coming from my LeNeT lab which has proven itself.
To train the model, I used AdamOptimizer which is similar to GD optimizer but uses Adam algorithm that simply helps algorithm to converge faster compared with GDO.
I decided to keep batch size of 128 to prevent memory issues.
I decided on 30 Epochs after some trials. Throughout the EPOCHS, I saved my parameters if the accuracy of the last iteration is bigger than the previously saved parameters. In this way, if the flactuations in the accuracy causes drop in the last iterations, I still keep the parameters from the iteration that yields the maximum accuracy.
Learning rate is set to 0.0008 to lower the accuracy flactuations.
I also decided to use sigmoid as activation function to minimze the number of dead neurons. I've also observed, with ReLu, the accuracy suddenly dies after 5-6 EPOCHS (drops from 0.80-0.900 to 0.050-0.060) which was not the case with sigmoid.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

As mentioned above, I've used the architecture from the LeNeT lab to start with. It has 3 convolutional layers + 3 fully connected layers.

1- I've seen a significant drop in the accuracy during the training (after 5-6 EPOCHS drops from 0.80-0.900 to 0.050-0.060) I suspected from underfitting.

2- I decided to increase the number of filter layers to increase the efficiency and prevent underfitting.

3- As I did not want to explode the memory and/or computational effort, I start increasing the number of filter depth, hence the size of the entire network.

4- After a number of iterations, accuracy reached the expected levels, however, I observed some flactutation that was preventing a reliable accuracy above 0.93. I decided to save the parameters for each iteration that yields a better accuracy.

My final model results were:
* training set accuracy of 
* validation set accuracy of 0.933 
* test set accuracy of 0.898


 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


