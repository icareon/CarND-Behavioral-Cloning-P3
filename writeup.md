#Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists the Nvidia structure which is a convolution neural network with 3x3 filter sizes and depths between 100 and 1 (model.py lines 89-106) 

The model includes ELU layers to introduce nonlinearity (code lines 93,95,96), and the data is normalized in the model using a Keras lambda layer (code line 90). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 94,97,99). 

The model was trained and validated on a large augmented data set to ensure that the model was not overfitting (code line 58-81). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 108).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to take the data, mirror it, normalize it, augment it and use images from three camera angles.

My first step was to use a convolution neural network model similar to the Nvidia network. I thought this model might be appropriate because it had been proven in the past to work well on data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it had dropout layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I balanced and added curve driving to the data. This resulted in the following distribution

![balanced histogram](./balanced.png)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 

- Normalization layer
- Cropping layer
- Convolution layer with 24 filters and 5x5
- Convolution layer with 36 filters and 5x5
- Convolution layer with 48 filters and 5x5
- Convolution layer with 64 filters and 3x3
- Convolution layer with 64 filters and 3x3
- A flatten layer
- Fully connected layer with 100 nodes
- Fully connected layer with 50 nodes
- Fully connected layer with 10 nodes
- Fully connected layer with 1 node


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the udacity provided data set.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the excellent behavior when driving the model. I used an adam optimizer so that manually training the learning rate wasn't necessary.

This resulted in a learning rate as shown here

![image1](./loss.png) "Model Visualization"
