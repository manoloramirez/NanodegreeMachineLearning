import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import IPython.display as display
from PIL import Image
import numpy as np
import json
import time
import glob
import argparse

import warnings
warnings.filterwarnings('ignore')


def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (img_size, img_size))/255.0
    return image.numpy()

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    expanded_image = np.expand_dims(image, axis=0)
    prediction = model.predict(expanded_image)
    probabilities, classes = tf.math.top_k(prediction, top_k)
    
    probabilities = probabilities.numpy()
    classes = classes.numpy()
    
    return probabilities, classes, image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('model')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names',default='label_map.json')
    args = parser.parse_args()
    
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
        
    path = args.image  
    trained_model = tf.keras.models.load_model(args.model,
                                       custom_objects={'KerasLayer':hub.KerasLayer})
    top_k = args.top_k
    
    probs, classes, image = predict(path, trained_model, top_k)
    
    print('Predicted flower name: \n',classes)
    print('Probabilities: \n ', probs)