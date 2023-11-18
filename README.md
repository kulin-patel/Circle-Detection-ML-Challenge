# Circle-Detection-ML-Challenge

Solution to the Challenge posted by SlingShot AI.

https://slingshotai.notion.site/Circle-Detection-ML-Challenge-070b2947a4334c2bab4378f821954cdd

# Challenge Description:

The goal of this challenge is to develop a **circle detector** that can find the location of a circle in an image with arbitrary noise. Your goal is to produce a deep learning model that can take an image as input and produce as output the location of the circle’s center (`x,y`) and its radius.

To get you started, we’ve provided some helper functions for generating example images. You can use the `noisy_circle` function to generate a square image with a circle randomly drawn on it and with added noise, along with the parameters `(x,y,r)`. The  `show_circle` function can be used to display a circle and see what it looks like. The `generate_examples` function provides an infinite generator with examples and corresponding parameters, and the `iou` function can be used to calculate the “[intersection over union](https://en.wikipedia.org/wiki/Jaccard_index)” (IOU) metric between the true location of the circle in the image and the predicted location.

As a test metric, we recommend accuracy based on thresholded IOU, because of its intuitive nature. For example, you can calculate the % of test examples where the predicted circle overlaps pretty closely with the actual circle - i.e. where IOU ≥ 0.5, 0.75, 0.9, or 0.95.

This project should ideally take around two hours (other than training time), with some experience training CNNs. We recommend using Colab for GPU training. A model with ~10M parameters is probably enough, although you can probably achieve high accuracy with substantially fewer. If you find it helpful, you’re welcome to play around with lower (or higher) noise parameters or starting with smaller images for faster training times.

Along with the model weights and code, please provide a short report, which can be in the form of a `README.md` explaining how the code works and the model’s final metrics. We care a lot about code quality, so we definitely recommend formatting your code nicely (e.g. using JetBrains or `black`), and using typing and/or comments where it’s helpful.


# Solution
This repository contains code to demonstrate the solution to the challenge.
The model is a simple CNN with 3 convolutional layers and 2 fully connected layers.

Purpose is to show coding style and ability to train a model to solve the problem.


## 1. Project Structure
1. `main.py`:
The entry point of the project. It reads the configuration, initializes the dataloaders, model, and trainer, and then starts the training process.
2. `dataset.py`:
Contains the CircleDataset class used for creating datasets for training, validation, and testing.
3. `model.py`:
Defines the CircleDetector class, the neural network model for circle detection.
4. `trainer.py`:
Contains the Trainer class for training the model, validating, and evaluating its performance.

Othe files:
- `circle_utils.py` - Contains the code for helper functions.
- `config.json` - Contains the configuration for training the model.

### Features
- **Configurable**: Easily adjust model and training parameters through the configuration file.
- **Model Training and Evaluation**: Comprehensive training and validation loops with performance logging.
- **Custom Dataset Handling**: Specialized dataset class for handling circle detection tasks.



## 2. CircleDataset: Circle Detection in Images
The CircleDataset class is a crucial component of the Circle Detection ML Challenge.
It's a custom dataset class designed for generating images with circles, complete with corresponding parameters.
This class extends` torch.utils.data.Dataset`, making it compatible with PyTorch's data loading and processing utilities.

### Features
- **Dynamic Circle Generation:** The dataset dynamically generates images with circles of varying radii and noise levels.
- **Configurable Parameters**: The size of the images, the range of circle radii, and the noise level can be configured via a configuration file.
- **Mode Selection**: Supports different modes for training, validation, and testing, allowing flexibility in how the dataset is utilized.
- **Compatibility with PyTorch DataLoader**: Easily integrates with PyTorch's DataLoader for efficient batch processing and data shuffling.

### Usage
The class is initialized with a configuration dictionary and a mode ('train', 'val', or 'test'). The configuration dictionary should include:

`img_size (int)`: The size of the images.
`min_radius (int)` : Minimum radius of circles.
`max_radius (int)`: Maximum radius of circles.
`noise_level (float)`: Noise level for the image.
`num_train_examples (int)`: Number of training examples.
`num_val_examples (int)`: Number of validation examples.
`num_test_examples (int)`: Number of test examples.
`device (str)`: Device to use for tensor ('cpu' or 'cuda').

## 3. CircleDetector: Neural Network for Circle Detection
- The` CircleDetector` class is a neural network model specifically tailored for detecting circles in grayscale images.
- This model, built using PyTorch's neural network module (nn.Module), is designed for a regression task.
- Its objective is to predict the parameters of a circle (x-coordinate, y-coordinate, and radius) within an image that includes noise.

### Model Overview
- **Purpose**: To detect and predict the parameters of circles in noisy grayscale images.
- **Model Type**: Convolutional Neural Network (CNN) designed for regression tasks.
- Features
- **Configurable Input Shape**: The model accepts an input shape parameter, allowing flexibility in handling various image sizes.
- **Deep Learning Layers**: Includes multiple convolutional layers, batch normalization layers, pooling layers, and fully connected layers for effective feature extraction and regression.
- **Activation Function**: Uses Leaky ReLU activation functions for non-linear transformations.


### Dynamic implementation Details
- `_calculate_fc_input_size Method`: Dynamically calculates the input size for the first fully connected layer based on the input shape.
- Extending and Customizing: The CircleDetector class can be extended or modified to include additional layers, different types of layers (like dropout for regularization), or to adapt to different image sizes and types of shapes.


## 3. Trainer: Training and Evaluation for Circle Detection
- The Trainer class is a comprehensive training utility designed for models, specifically in the context of circle detection in images.
- It encapsulates the entire training process, including model evaluation, saving checkpoints, and plotting training and validation losses.

### Features
- **Configurable Training Process**: Supports various configurations for training, including optimizer and scheduler settings.
- **Evaluation Metrics**: Implements metrics such as mean Intersection Over Union (IOU) and thresholded IOU for model validation and evaluation.
- **thresholded IOU** =  dictionary containing IOU values for a range of thresholds (0.5, 0.75, 0.9, 0.95).
- **Checkpoint Management**: Ability to save and load model checkpoints, allowing for resuming training and evaluating pre-trained models.
- **Loss Plotting**: Functionality to plot training and validation loss, aiding in visual analysis of the model's performance.

### Model Training Lifecycle
-** Initialization**: The Trainer class is initialized with configurations, model, and data loaders.
- **Optimizer and Scheduler Creation**: Based on the provided configurations, the trainer sets up the optimizer and learning rate scheduler.
- **Training and Validation Loop**: The model is trained for a specified number of epochs, validating and saving checkpoints as configured.
- **Model Evaluation**: Evaluate the model on the test set using metrics like loss and IOU.
- **Checkpoint Management**
- **Loss Visualization**: Plot the training and validation loss over epochs.

### Implementation Details
- `create_optimizer` and `create_scheduler Methods`: For initializing the optimizer and scheduler based on configurations.
- `train`,` train_epoch`, and `train_step Methods`: To carry out the training process.
- `validate` and` evaluate` Methods: For model validation and evaluation.
- `get_metrics` Method: To calculate evaluation metrics.
- `predict` Method: For making predictions using the trained model.
- `load_checkpoint` and `save_checkpoint` Methods: For managing model checkpoints.
- plot Method: To visualize the training and validation loss.


## Note on Modification of Intersection Over Union (IOU) Function
In the original Circle Detection ML Challenge, the Intersection Over Union (IOU) function was a key component, originally implemented using NumPy. I've modified this function for three main reasons:

1. **Integration with PyTorch**: The original function used NumPy, which was less efficient for tensor operations. I reimplemented it using PyTorch to optimize performance and compatibility with GPU acceleration.

3. **Accuracy for Enclosed Circles**: The initial IOU calculation incorrectly returned a value of one when one circle was entirely within another. This was an important flaw, especially in cases with smaller circles enclosed by larger ones. The new implementation corrects this, ensuring accurate IOU calculations for enclosed circles.

5. **General Case Accuracy:** Apart from enclosed circles, the original IOU formula had inaccuracies in various scenarios. My modification ensures more precise IOU values across a wider range of cases, improving the model's reliability and the validity of its performance evaluation.