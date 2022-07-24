# Learning-MNIST-just-in-1-epoch
## An efficient CNN for learning MNIST

This is a very small and efficient model that can learn MNIST very fast. Convolutional neural networks(CNNs) are used for training.
only one epoch is used for training this model.

### Architecture:
Two blocks of CNNs followed by a fully connection layer(FC) are used to learn MNIST dataset.

![MNIST Architecture](https://user-images.githubusercontent.com/106428795/180659712-7ccb5420-2f23-4cd9-9fd4-0e588ad2e75a.jpg)

Each block consists of a Convolution layer and a max pooling to decrease the size of previous layer.

For training we used two more layers: Drop out layer and Batch normalization.

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

![20200110092811](https://user-images.githubusercontent.com/106428795/180656637-5bbc3010-9511-470d-97d3-d66cf54f768b.jpg)

### Accuracy(98%):

This model is designed to learn very fast. we trained it just one epoch and then checked the accuracy of it by giving all test samples.

It could reach 98% after one epoch!
