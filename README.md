# Siamese-Network-with-Triplet-Loss

This project contains two sections.

* The first part uses a parallel feature model to prodeuce an embedding representation of the Mnist dataset the model is trained using triplet loss this function aims to force the image towards other images in it's class and further away from images in other classes.

* The second part uses a pre trained network, more specficaly a fine tuned version of the VGG16 convolutional network trained on facial images to produce an embeddind representation of a number of celebrity faces to produce a simple retrival system.

## Part One.

A triplet loss network was implemented in Python using the Keras framework and a skeleton file provided by Dr. McDermott that demonstrated the structure and methodology of a triplet loss network. This code already provided functions to load the MNIST dataset and generate triplet batches. However to complete the network, an embedding model and a loss function had to be created. A simple embedding model was created that received a single 28x28 RGB image and outputted a dense vector of length ten. The model only contained two convolutional layers as a sufficient accuracy level was achieved.

<p align="center">
  <img width="460" height="300" src="images/image1.png">
</p>

Using the lecture notes; the equation for the triplet loss was translated to a Keras tensor friendly format. This function minimizes the loss between embedding’s of a similar class by minimizing  the distance from the anchor to positive embedding whilst maximizing distance from the negative and anchor. The alpha/margin value makes sure that the network is not allowed to output the trivial solution where all embedding’s vectors are zero or contain the same values. The impact of this parameter can be seen in figure (1) where the alpha value is varied from 0 to 1. From this figure the resulting embedding’s improve from the trivial solution as the margin is increased. The margin essentially defines a minimum threshold between the positive and the negative images. Thus as the margin is increased the total number of triplets generated whose loss is actually greater than zero decreases therefore they do not contribute to the training of the model thus reducing the accuracy of the outputted embedding’s. Also with higher alpha values the embedding clusters are created much tighter however the clusters themselves are more densely packed together. Despite this a value of 0.4 was chosen for the remainder of the assignment. Below is the code for the triplet loss function implemented in keras.

<p align="center">
  <src="images/image2.png">
</p>
