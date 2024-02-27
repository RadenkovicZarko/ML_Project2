# Machine Learning Project

This repository contains the source code and dataset for a series of machine learning challenges focusing on decision trees, neural networks, and behavioral cloning. Below you will find the details of each problem and the objectives we aim to achieve.

## Problem 1: Decision Trees 

### Objective
Train a decision tree model to classify Titanic passengers as survivors or non-survivors based on provided training and test datasets.

### Dataset
train.csv: Training data.
test.csv: Test data.
A detailed description of the dataset and the objective of the research can be found link to the dataset description](https://www.kaggle.com/competitions/titanic).

### Requirements
Preprocess and organize data as necessary for optimal model performance.


## Problem 2: Neural Network 

### Objective
Implement and train a neural network model to classify Titanic passengers as survivors or non-survivors using the same dataset as in Problem 1.

### Implementation
The neural network must be implemented manually in matrix form without using the Keras library.
Data preprocessing and organization are allowed and expected as needed.


## Problem 3: Behavioral Cloning [40 points]

### Objective
Implement behavioral cloning through a car simulator, where the model learns to drive a car autonomously on a track.

### Components
Simulator: 3 versions for different operating systems are provided for creating (recording) datasets and testing the trained model.

### Python Files:
utils.py: Contains helper functions, a variable with the input image size (INPUT_SHAPE), and a batch_generator function.
drive.py: Used for testing the trained model.
model.py: Executes the training process. Fill in the missing code in build_model and train_model functions.


### Training and Testing
Data Collection: Use the simulator's Training Mode to record data. Manually drive the car to collect sufficient training data.
Model Training: Run model.py with the collected data to train your model. Use Keras model and ModelCheckpoint for model saving during training.
Testing: Use drive.py to test the trained model in the simulator's Autonomous Mode.



## Getting Started
Clone this repository to your local machine.
Ensure you have Python installed and create a virtual environment.
Follow the instructions in each problem's section to train and test the models.
