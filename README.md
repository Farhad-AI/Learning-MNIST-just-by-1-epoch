# Learning-MNIST-just-in-1-epoch
## An efficient CNN for learning MNIST

This is a very small and efficient model that can learn MNIST very fast. Convolutional neural networks(CNNs) are used for training.
only one epoch is used for training this model.

### Architecture:
Two blocks of CNNs followed by a fully connection layer(FC) are used to learn MNIST dataset.


Each block consists of a Convolution layer and a max pooling to decrease the size of previous layer.


For training we used two more layer: Drop out layer and Batch normalization.

      Drop out helps our network against overfitting, so the model can learn more!      
      Batch normalization is useful in order to remove bias of layers before we use them for feeding the next layer.     
##### But we remove both Drop_out and Batch_normalization when we the training process is finished, and we want to evalute it on test_data.
It is easy! For disabling them we just need to put "model.eval()" before testing our model and for enabling them using "model.train()".
      
      self.conv1 = nn.Sequential(
                                  nn.Conv2d(1, 5, 3, 1),
                                  nn.ReLU(),
                                  nn.Dropout2d(p=0.1),
                                  nn.MaxPool2d(2, 2),
                                  nn.BatchNorm2d(5) 
                                        )
### Loading the data
It is easy to load the data by using the commands that PyTorch prepared for us.

      train_data = datasets.MNIST(root="MNIST/", train=True, download=True, transform=transform)
      test_data = datasets.MNIST(root="MNIST/", train=False, download=True, transform=transform)
there are 60000 samples for training and 10000 samples for testing.
### Accuracy:
This model is designed to learn very fast. we learned it just one epoch and check the accuracy of it by giving all the test data.

It could reach # 98% by just one epoch!
