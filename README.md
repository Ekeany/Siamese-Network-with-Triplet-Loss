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
  <img width="393" height="147" src="images/image2.png">
</p>


The model was then compiled and trained on the MNSIT dataset using the batch generator provided. Below is a graph of the training and validation losses as a function of epoch. 

<p align="center">
  <img width="502" height="357" src="images/image3.png">
</p>

In order to create a recognition function, that not only distinguishes between the classes in the MNIST dataset but also classifies if an image is an integer or a digit, a new model had to created. The problem with the previous model was that it had an input of three RGB images and it’s triplet loss function didn’t use the class labels to update the gradients. 
To overcome the first problem; and output a single embedding for a single image, the weights of the model had to be saved and uploaded to a new model that had the same structure as the embedding model but with a input dimension that corresponded to a single image and not a triplet. Below is the implementation of this new embedding model.

<p align="center">
  <img width="567" height="189" src="images/image4.png">
</p>

5000 images where then passed through this pre-trained network and their embedding’s were calculated. Using dimensionality reduction a 2d visualization of the embedding’s was created where the color and text indicate the class of a point or cluster. From the visualization below we can see that the model has been trained correctly, as embedding’s of the same class have formed clusters in vector space.  

<p align="center">
  <img width="433" height="440" src="images/image5.png">
</p>

A clustering algorithm would be able to split the image into 9 clusters however it would be unable to distinguish the label of a point as it is an unsupervised learning technique. Thus to compute the label of each embedding a simple classifier was created that accepted an embedding vector and translated this to a one hot encoded label. As the embedding vectors have already encompassed the class of each image the model was extremely simple containing just an Input and a softmax output layer. After training the model with the target one hot encoded vectors an accuracy level of 88% was achieved.
