#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center_lane_driving]: ./images/center_lane_driving.jpg "Center Lane Driving"
[original_image]: ./images/original_image.jpg "Original Image"
[recover_left1]: ./images/recover_left1.jpg "Recover Left 1"
[recover_left2]: ./images/recover_left2.jpg "Recover Left 2"
[cropped_image]: ./images/cropped_image.jpg "Cropped Image"
[flipped_image]: ./images/flipped_image.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model architecture is as follows (refer to End to End Learning for Self-Driving Cars):
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 90, 320, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 3)    228         lambda_1[0][0]                   
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 21, 79, 3)     0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 21, 79, 3)     0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 21, 79, 3)     0           activation_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 9, 38, 24)     1824        dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 9, 38, 24)     0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 9, 38, 24)     0           activation_2[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 3, 17, 36)     21636       dropout_2[0][0]                  
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 1, 15, 48)     15600       convolution2d_3[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 720)           0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 720)           0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           72100       dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 116959
____________________________________________________________________________________________________
```

####2. Attempts to reduce overfitting in the model

The model contains dropout layers (after the 1st, 2nd convolutional layers, after the flatten layer) in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 34-35-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The learning rate was manually tuned to 0.0001.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use transfer learning.

My first step was to use a convolution neural network model similar to the Nvidia end-to-end learning network. I thought this model might be appropriate because Nvidia has successfully use the model on real driving datasets.

Firstly, I only trained the model for 5 epoch to save time. The vehicle could not pass the fork formed by tires. Then, I tried to trained for more epoch. The vehicle could eventually pass the fork. Then I slightly modified the network architecture to try to train a model in only 5 epoch. Finally, the best model is trained for 10 epoch to ensure the vehicle follow the centerline.

To combat the overfitting, I modified the model by adding dropout layers.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. If trained for more epoch, the vehicle could follow the centerline. As for track 1, the speed could be set to 30. However, the speed could only be set to 15 for track 2. More recovery data from track 2 may increase the speed.


####2. Creation of the Training Set & Training Process

The data quality proved to be essential for the training. To exclude the influence of bad dataset, only the dataset provided by udacity was used for training as I assumed the dataset provided by udacity is better than dataset sampled by myself. After I successfully trained a model, I tried to train a new model based on the dataset sampled by myself. 

The dataset provided by udacity is captured by joystick as the steering angle change is much more slowly. The dataset captured by keyboard suffers from sharp change.

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center_lane_driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from side drive. These images show what a recovery looks like starting from left :

![alt text][recover_left1]
![alt text][recover_left2]

Then I repeated this process on track two in order to get more data points.

To get more samples, all the 3 cameras were used to sample data. A coefficient was added to compensate for the camera position.

To augment the data sat, I also flipped images and angles thinking that this would increase the diversity of the sample data. For example, here is an image that has then been flipped:

![alt text][cropped_image]
![alt text][flipped_image]

After the collection process, I had 83472 (13912x3x2) number of data points. I then preprocessed this data by cropping the top 50 and bottom 20 rows of pixels.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the loss change.
