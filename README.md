# Image-Classifier-Project : First AI application 

### Projects details :

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, I will train an image classifier to recognize different species of flowers. I can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice I'd train this classifier, then export it for use in your application. We'll be using this <a href="https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html">dataset</a> from Oxford of 102 flower categories, you can see a few examples below. 

![Screenshot 2022-10-19 at 14 31 10](https://user-images.githubusercontent.com/74813723/197958990-636b664e-929d-4905-b32c-41ac689dd6e3.png)

## Project overview :
The project is broken down into multiple steps:

- Load the image dataset and create a pipeline.
- Build and Train an image classifier on this dataset.
- Use your trained model to perform inference on flower images.
- We'll lead you through each part which you'll implement in Python.

### Install Datasets and Upgrade TensorFlow
To ensure we can download the latest version of the oxford_flowers102 dataset, let's first install both tensorflow-datasets and tfds-nightly.
tensorflow-datasets is the stable version that is released on a cadence of every few months, tfds-nightly is released every day and has the latest version of the datasets We'll also upgrade TensorFlow to ensure we have a version that is compatible with the latest version of the dataset.

## Load the Dataset
Here you'll use <a href="https://www.tensorflow.org/datasets/catalog/oxford_flowers102">tensorflow_datasets to load the Oxford Flowers 102 dataset </a>. This dataset has 3 splits: 'train', 'test', and 'validation'. You'll also need to make sure the training data is normalized and resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet, but you'll still need to normalize and resize the images to the appropriate size.

## Build and Train the Classifier¶
Now that the data is ready, it's time to build and train the classifier. I should use the MobileNet pre-trained model from TensorFlow Hub to get the image features. Build and train a new feed-forward classifier using those features. Follows step:
- Load the MobileNet pre-trained network from TensorFlow Hub.
- Define a new, untrained feed-forward network as a classifier.
- Train the classifier.
- Plot the loss and accuracy values achieved during training for the training and validation set.
- Save your trained model as a Keras model.
- Inference for Classification
- Write your Algorithm to web app

# Building the Command Line Application
Now that I've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. Your application should be a Python script that run from the command line. For testing, I should use the saved Keras model I saved in the first part.
The predict.py file that uses a trained network to predict the class for an input image. The predict.py module should predict the top flower names from an image along with their corresponding probabilities.

### Basic usage (run this app):
$ python predict.py --input /path/to/image --model saved_model --top_k K --category_names map.json

Options:
- input : path image
- top_k : Return the top K most likely classes
- model : save model
- category_names : Path to a JSON file mapping labels to flower names:

### Examples
For the following examples, we assume we have a file called orchid.jpg in a folder named test_images that contains the image of a flower. We also assume that we have a Keras model saved in a file named my_model.h5.

- $ python predict.py --input./test_images/orchid.jpg --model my_model.h5 --top_k 3 --category_names label_map.json

## Images for Testing
We have provided 4 images in the test_images folder in order to check my predict.py module. 
* The 4 images are:

- cautleya_spicata.jpg
- hard-leaved_pocket_orchid.jpg
- orange_dahlia.jpg
- wild_pansy.jpg

# Results
Please look my notebook and predict.py file for all details for this project


