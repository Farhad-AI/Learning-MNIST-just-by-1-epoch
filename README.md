# Learning-MNIST-just-in-1-epoch
## An efficient CNN for learning MNIST

This is a very small and efficient model that can learn MNIST very fast. Convolutional neural networks(CNN) are used for training.
only one epoch is used for training this model.

### Architecture:
Two blocks of CNNs followed by a fully connection layer(FC) are used to learn MNIST dataset.
Each block consists of a Convolution layer and max pooling to decrease the size of previous layer.
For training we used two more layer: Drop out layer and Batch normalization.
      Drop out helps our network from overfitting, so the model can learn more
      Batch normalization is useful in order to remove bias of layers before we use them for feeding the next layer.
