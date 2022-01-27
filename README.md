# Aircraft Pitch Prediction Challenge
This project is developed to compete in the kaggle [Aircraft Pitch Prediction Challenge](https://www.kaggle.com/c/pitch-aileron/overview). The task is to develop a program to predict the pitch (or vertical heading) of the aircraft given various parameters based on readings made by the aircraft's instruments. 

## Project structure
The repository contains the following files/folders:
* **data/** is the folder containing the dataset already splitted in traing set (train.csv) and test set (test.csv)
* **pitch_aileron.ipynb** is the colab notebook where the models are defined and trained 

## Frameworks and Libraries
The project is developed using Python programming language on Google Colaboratory platform. The task is performed as a regression problem. It is solved using a Deep Neural Network (DNN) exploiting Keras API on top of Tensorflow framework. 

## Workflow
### Data preprocessing
I preprocessed the dataset by doing a normalization of the data applying a max-min scaling to suppress the effect of features with high magnitude. I split the training set to have a test set on which to calculate evaluation metrics.

### Training
To better solve the regression I build and train different models. It is important to introduce some metrics to have a fair comparison: R-squarred and Mean Absolute Error (MAE). I train 5 models by varying some hyperparameters such as the number of layers, number of neurons per layer, activation function, etc. To avoid redundancy in the documentation I describe only the model I use to compute the prediction of the challenge. I train the model after setting an early stopping condition that monitors the loss function in order to avoid some overfitting phenomena. I use *Adam* as optimizer.

### DNN architecture
The DNN architecture is composed by 3 dense layers with the same activation function: tanh. Between the dense layers there is a Dropout layer (rate = 0.1). 
```
Model: "regressor"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_20 (InputLayer)       [(None, 16)]              0         
                                                                 
 dense_56 (Dense)            (None, 128)               2176      
                                                                 
 dropout_33 (Dropout)        (None, 128)               0         
                                                                 
 dense_57 (Dense)            (None, 64)                8256      
                                                                 
 dropout_34 (Dropout)        (None, 64)                0         
                                                                 
 dense_58 (Dense)            (None, 16)                1040      
                                                                 
 dropout_35 (Dropout)        (None, 16)                0         
                                                                 
 dense_59 (Dense)            (None, 1)                 17        
                                                                 
=================================================================
Total params: 11,489
Trainable params: 11,489
Non-trainable params: 0
_________________________________________________________________
```

The DNN obtains the following metrics: R^2 = 0.8, MAE=0.0021
