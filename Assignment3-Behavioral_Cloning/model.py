import csv
import cv2
import os
#open the csv file
samples=[]
with open('C:\\Users\\nxa04630\\Desktop\\data\\data\\driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #extract the data line by line
    for line in reader:
        samples.append(line)
        
# delete the first row
del(samples[0])

# sanity check: number of samples, and log file content
print(len(samples))
print(samples[2])

# import libraries and split the dataset (0.8-0.2)
import numpy as np
import sklearn
from random import shuffle
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# angle correction based on trial-error
angle_correction = 0.15

# data generator for training
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                ## center image
                source_path = batch_sample[0]
                path = source_path.split('/')[-1]
                name = 'C:/Users/nxa04630/Desktop/data/data/IMG/' + path
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                ## flipped center image
                flipped_img = cv2.flip(center_image, 1)
                images.append(flipped_img)
                measurement_flipped = -center_angle
                angles.append(measurement_flipped)
                
                # side camera - right
#                 name_right = batch_sample[2].split('/')[-1] -- for own samples
                source_path = batch_sample[2]
                path = source_path.split('/')[-1]
                name_right = 'C:/Users/nxa04630/Desktop/data/data/IMG/' + path
                right_cam_image = cv2.imread(name_right)
                images.append(right_cam_image)
                right_cam_angle = center_angle - angle_correction
                angles.append(right_cam_angle)
                
                # side camera - left
#                 name_left = batch_sample[1].split('/')[-1]
                source_path = batch_sample[1]
                path = source_path.split('/')[-1]
                name_left = 'C:/Users/nxa04630/Desktop/data/data/IMG/' + path
                left_cam_image = cv2.imread(name_left)
                images.append(left_cam_image)
                left_cam_angle = center_angle + angle_correction
                angles.append(left_cam_angle)

           
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# create a regression network
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt

model = Sequential()

# normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
print('After Normalization')
print(model.layers[-1].output_shape)

# trim image to only see section with road
#cropping 80 from top, 25 from bottom, 0 left, 0 right
model.add(Cropping2D(cropping=((80,25),(0,0))))
print('After Cropping')
print(model.layers[-1].output_shape)

# conv1 -> 24 of 5x5 filters with 2x2 strides followed by ReLu activation layer
model.add(Convolution2D(24, (5, 5), activation='relu', strides=(2,2)))
print('After conv1')
print(model.layers[-1].output_shape)

# dropout1 to reduce overfitting
model.add(Dropout(0.1))
print('After dropout')
print(model.layers[-1].output_shape)

# conv2
model.add(Convolution2D(36, (5, 5), activation='relu', strides=(2,2)))
print('After conv2')
print(model.layers[-1].output_shape)

# conv3
model.add(Convolution2D(48, (5, 5), activation='relu', strides=(2,2)))
print('After conv3')
print(model.layers[-1].output_shape)

# dropout2
model.add(Dropout(0.2))

# conv4
model.add(Convolution2D(64, 3, 3, activation='relu'))
print('After conv4')
print(model.layers[-1].output_shape)

# conv5 
model.add(Convolution2D(64, 2, 2, activation='relu'))
print('After conv5')
print(model.layers[-1].output_shape)

# 3 fully connected layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

from keras.models import Model
import matplotlib.pyplot as plt
batch_size = 32

# model to minimze the loss with means squared error function as this is a regression network
# adam optimizer
model.compile(loss='mse', optimizer='adam')

# train and store the object data
history_object = model.fit_generator(train_generator, steps_per_epoch= (len(train_samples)/batch_size), validation_data=validation_generator, 
                    validation_steps=(len(validation_samples)/batch_size), nb_epoch=5, verbose=1)

# save model
model.save('model_udacity.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

######################################################################################################
### If further training needed with more datapoints, uncomment next section -> load model and train ##
######################################################################################################

##restoring the model
#from keras.models import load_model
#model = load_model('model_udacity.h5')

# keep training the loaded model
# model.compile(loss='mse', optimizer='adam')
# history_object = model.fit_generator(train_generator, steps_per_epoch= (len(train_samples)/batch_size), validation_data=validation_generator, 
#                     validation_steps=(len(validation_samples)/batch_size), nb_epoch=3, verbose=1)

# model.save('model_udacity.h5')
# ### print the keys contained in the history object
# print(history_object.history.keys())

# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()
# # first decrease then increase ==>> overfitting the training data
