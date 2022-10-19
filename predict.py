# import libraries
import numpy as np
import json
import tensorflow as tf
import argparse
import tensorflow_hub as hub
from PIL import Image
import os
# Avoid tensorflow print on standard error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure the input data of model
parser = argparse.ArgumentParser(description='Short sample  AI app for flowers classifier')
parser.add_argument('--input', default='./test_images/cautleya_spicata.jpg', action="store", type = str, help='Image path')
parser.add_argument('--model', default='./my_model.h5', action="store", type = str, help='Classifier path')
parser.add_argument('--top_k', default=5, action="store", type=int, help='Return the top K most likely classes')
parser.add_argument('--category_names', default='./label_map.json', action="store", type=str, help='Path to a JSON file mapping labels to flower names')

arg_parser = parser.parse_args()
image_path = arg_parser.input
model_path = arg_parser.model
top_k = arg_parser.top_k
category_names = arg_parser.category_names

def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image=image.numpy()
    return image


def predict(image_path, model, top_k):
    """predict class"""
    if top_k < 0:
        print('Error, top_k must be >=0')
        exit()

    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    expanded_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(expanded_image)
    probs, indices = tf.math.top_k(predictions, k=top_k)
    
    probs = list(probs.numpy()[0])
    labels = list(indices.numpy()[0])
    return probs, labels


if __name__== "__main__":
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)

    reloaded_keras_model = tf.keras.models.load_model(model_path,custom_objects = {'KerasLayer':hub.KerasLayer},compile=False)
    probs, labels = predict(image_path, reloaded_keras_model, top_k)

    print ("\n*** Top {} Classes ***\n".format(top_k))

    for i, prob, label in zip(range(1, top_k+1), probs, labels):
        print(i)
        print('Label:', label)
        print('Class name:', class_names[str(label+1)])
        print('Probability:', prob)
        print('----------')