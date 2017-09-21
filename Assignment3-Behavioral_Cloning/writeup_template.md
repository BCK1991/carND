#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Curve1.png "Curve1"
[image2]: ./Curve2.png "Curve2"
[image3]: ./Curve3.png "Curve3"
[image4]: ./Curve4.png "Curve4"
[image5]: ./model.png "Visualization of the architecture"
[image6]: ./CenterImg.png "CenterImg"

[image8]: ./LeftImg.png "Left Cam Image"
[image9]: ./RightImg.png "Right Cam Image"
[image10]: ./OrgImg.png "Original Image"
[image11]: ./FlippedImg.png "Flipped Image"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_udacity.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video of successful run in autonomous mode

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_udacity.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed (from NVIDIA paper: 'End to End Learning for Self-Driving Cars')

My model consists of a convolution neural network with 3 5x5 filter sizes and depths between 24, 36, and 48. Each followed by ReLu activation layer.  (model.py lines 18-24) 

Then two convolutional layers: 3x3 and 2x2 with depths of 64. 

Finally, 3 fully connected layers are added.

The data is normalized in the model using a Keras lambda layer (code line 96). 

To reduce the data size and eliminate the uninteresting features from the images, Keras cropping layer added. It crops the images 80 pixels from top (trees and other landscapes), 25 pixels from bottom (hood of the car).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 112 and 127). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 177-198). The dataset has been gradually increased based on the simulation results: if car goes out-of-the track at a certain curve, I've added extra data by using the simulator at that specific curve. Here are some examples of difficult curves that required extra training data.

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image3]

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 152).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center camera and flipped center camera images to generalize, and also added the left and right sides camre images with angle correction to recover from lane departures. 

For details about how I develop the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a proven architecture from NVIDIA, mentioned above.

My first step was to use a convolution neural network model similar to the NVIDIA End-to-end driving paper, I thought this model might be appropriate because the concept explained in the paper is very similar with our assignment objectives.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added dropout layers.

Then I collected more data points.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, see the images above (difficult curves), to improve the driving behavior in these cases, I captured more data from only the failing part of the track.

By using this iterative way, I've reached to 18k images (before augmentation, so after the number reaches 54k). However, I could not make the car go through the curve 3. 

I tried to fine-tune the steering offset for left and right side images, it helped the car to center the road a bit better, however, I was still having difficulties in the curves.

I went back to the provided (udacity) dataset and trained my algorithm for 5 epochs.

I tested my architecture

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

I compared the behavior of the car after training with dataset that I collect (model.h5) and after training with udacity dataset (model_udacity.h5). I saw while driving with model.h5, the steering angle was changing rapidly and drastically (too many swerves) although keeping the car in the road was not a problem, the car was not really staying in the center. When I use the udacity dataset, on the other hand, the amount of change in the steering angle was smoother and less frequent. My first estimate of reason for the unexpected behavior is the streeing data captured by myself: as I was using the keyboard, I needed to use rapid key strokes to keep the car in the middle of the road (as keyboard gives you a bang-bang control option for steering, either 0 or 25 (max)). Although, the avarage steering angle throughout the curve should be the same independent of the method, I could not really come-up with another explanation.

My conclusion is, the success of the network is dependent on the dataset as much as it is dependent on the architecture.

####2. Final Model Architecture

The final model architecture (model.py lines 93-144) consisted of a convolution neural network with the following layers and layer sizes 

* Convolution Layer 1: 5x5 filter of depth 24 (2x2 strides)
* Activation Layer 1: ReLu
* Dropout 1: 0.1
* Convolution Layer 2: 5x5 filter of depth 36 (2x2 strides)
* Activation Layer 2: ReLu
* Convolution Layer 3: 5x5 filter of depth 48 (2x2 strides)
* Activation Layer 3: ReLu
* Dropout 2: 0.2
* Convolution Layer 4: 3x3 filter of depth 64
* Activation Layer 4: ReLu
* Convolution Layer 5: 2x2 filter of depth 64
* Activation Layer 5: ReLu
* Fully connected 1-2-3 : 100 - 50 - 10


Here is a visualization of the architecture 

![alt text][image5]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image6]



I then also included left and right camera images so that the vehicle would learn to recovery when it goes out of the track. These images are from left and the right cameras:


![alt text][image8]
![alt text][image9]


To augment the data sat, I also flipped images and angles thinking that this would help generalizing the network. For example, here is an image that has then been flipped:


![alt text][image10]
![alt text][image11]


I kept collecting data especially for the parts of the circuit that are challenging. (That I've seen the network is failing)

After the collection process, I had 18k number of data points. I then preprocessed this data by lambda layer and cropped the un-interesting parts by using cropping in keras tool.

I finally randomly shuffled the data set and put 20Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the fact that after 5 epoches the loss starts fluctuating around the same value. I used an adam optimizer so that manually training the learning rate wasn't necessary.
