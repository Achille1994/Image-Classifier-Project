# Image-Classifier-Project

### Projects details :

- First AI application¶: Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. In this project, I will train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using this <a href="https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html">dataset</a> from Oxford of 102 flower categories, you can see a few examples below.

## Project overview :
The project is broken down into multiple steps:

- Load the image dataset and create a pipeline.
- Build and Train an image classifier on this dataset.
- Use your trained model to perform inference on flower images.
- We'll lead you through each part which you'll implement in Python.

### Install Datasets and Upgrade TensorFlow¶
To ensure we can download the latest version of the oxford_flowers102 dataset, let's first install both tensorflow-datasets and tfds-nightly.

tensorflow-datasets is the stable version that is released on a cadence of every few months
tfds-nightly is released every day and has the latest version of the datasets
We'll also upgrade TensorFlow to ensure we have a version that is compatible with the latest version of the dataset.
