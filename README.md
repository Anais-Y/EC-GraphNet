# An Effective Connectivity Driven Graph Convolutional Neural Network for EEG based Emotion Recognition

## Introduction 
This project implements the **EC-GraphNet**, a deep learning framework designed for modeling complex spatio-temporal dependencies of EEG singal. The framework combines graph convolution with self-attention mechanisms and is tailored for emotion prediction tasks.

## Main Modules
The project consists of three main components:
1. [train.py]
  - Handles training pipeline, including data loading, hyperparameter configuration, training, validation, and testing.
2. [engine.py]
  - Contains core training logic and optimization routines.
  - Implements features like learning rate decay, gradient clipping, and custom loss functions.
3. [model_permute.py]
  - Graph Convolution Module (gcn_operation)
  - Spatio-Temporal Graph Convolution Layers (STSGCL)
  - Spatio-Temporal Graph Convolution Modules (STSGCM)
  - Self-Attention Mechanism based classifier (output_layer)


## Acknowledgements

This project is inspired by [STSGCN_Pytorch](https://github.com/SmallNana/STSGCN_Pytorch).  
The code has been adapted and modified to suit the specific requirements of this project.
