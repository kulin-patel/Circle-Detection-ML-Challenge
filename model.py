''' Model class for the circle detection problem.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model class
class CircleDetector(nn.Module):
    """
    A neural network model for detecting circles in grayscale images.

    This model is designed for a regression task to predict the parameters of a circle (x, y, radius)
    drawn on a grayscale image with added noise.

    Parameters:
        input_shape (tuple, optional): The shape of the input tensor. Default: (1, 100, 100)

    """
    def __init__(self, input_shape=(1, 100, 100)):
        ''' Initialize the model.'''
        super(CircleDetector, self).__init__()

        channels_list = [1,128, 512, 64]
        kernel_size = [5,3,3]

        self.conv1 = nn.Conv2d(channels_list[0], channels_list[1], kernel_size[0])
        self.batchnorm1 = nn.BatchNorm2d(channels_list[1])
        self.conv2 = nn.Conv2d(channels_list[1], channels_list[2], kernel_size[1])
        self.batchnorm2 = nn.BatchNorm2d(channels_list[2])
        self.conv3 = nn.Conv2d(channels_list[2], channels_list[3], kernel_size[2])
        self.batchnorm3 = nn.BatchNorm2d(channels_list[3])

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Calculate the size for the fully connected layer
        fc_input_size = self._calculate_fc_input_size(input_shape)

        self.fc1 = nn.Linear(fc_input_size, 1024)
        self.fc2 = nn.Linear (1024, 128)
        self.fc3 = nn.Linear(128, 3)


    def forward(self, x):
        ''' Forward pass of the model.'''
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        x = self.pool(F.relu(self.batchnorm3( self.conv3(x) )))
        x = x.view(x.size(0), -1)  # Adjusted for the flattened size
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x

    def _calculate_fc_input_size(self, input_shape):
        ''' Calculate the input size for the fully connected layer.'''
        x = torch.randn(1, *input_shape)

        # Pass the dummy tensor through the convolutional and pooling layers
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))

        # The output size is the product of the dimensions of the output tensor
        return x.view(-1).shape[0]

# Model class
class CircleDetector2(nn.Module):
    """
    A neural network model for detecting circles in grayscale images.

    This model is designed for a regression task to predict the parameters of a circle (x, y, radius)
    drawn on a grayscale image with added noise.

    Parameters:
        input_shape (tuple, optional): The shape of the input tensor. Default: (1, 100, 100)

    """
    def __init__(self, input_shape=(1, 100, 100)):
        ''' Initialize the model.'''
        super(CircleDetector2, self).__init__()

        channels_list = [1,128, 512, 64]
        kernel_size = [5,3,3]

        self.conv1 = nn.Conv2d(channels_list[0], channels_list[1], kernel_size[0])
        self.batchnorm1 = nn.BatchNorm2d(channels_list[1])
        self.conv2 = nn.Conv2d(channels_list[1], channels_list[2], kernel_size[1])
        self.batchnorm2 = nn.BatchNorm2d(channels_list[2])
        self.conv3 = nn.Conv2d(channels_list[2], channels_list[3], kernel_size[2])
        self.batchnorm3 = nn.BatchNorm2d(channels_list[3])

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Calculate the size for the fully connected layer
        fc_input_size = self._calculate_fc_input_size(input_shape)

        self.fc1 = nn.Linear(fc_input_size, 1024)
        self.fc2 = nn.Linear (1024, 128)
        self.fc3 = nn.Linear(128, 3)


    def forward(self, x):
        ''' Forward pass of the model.'''
        x = self.pool1(F.leaky_relu(self.batchnorm1(self.conv1(x))))
        x = self.pool2(F.leaky_relu(self.batchnorm2(self.conv2(x))))
        x = self.pool3(F.leaky_relu(self.batchnorm3( self.conv3(x) )))

        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _calculate_fc_input_size(self, input_shape):
        ''' Calculate the input size for the fully connected layer.'''
        x = torch.randn(1, *input_shape)

        # Pass the dummy tensor through the convolutional and pooling layers
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))

        # The output size is the product of the dimensions of the output tensor
        return x.view(-1).shape[0]

if __name__ == '__main__':

    model = CircleDetector()
    print(model)
    print('Model Size: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
