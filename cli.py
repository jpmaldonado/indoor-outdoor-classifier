
import os
import cv2
import argparse

# Disable keras "Using Tensorflow backend" message
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from keras import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
sys.stderr = stderr

# Disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_model():
    # Initialize the saved model
    model = Sequential([
        Flatten(input_shape=(100, 100,3)),
        Dense(128, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
        ])
    model.load_weights("model.h5")
    return model

def preprocess_img(image):
    # Read and format the image to the model shape
    img = cv2.imread(image)
    img = cv2.resize(img,(100,100))
    img = img.reshape(1,100,100,3)
    return img

def main(image):
    model = load_model()
    img = preprocess_img(image)
    pred = model.predict(img)
    label = 'indoor' if pred==0 else 'outdoor' #Classes are stored alphabetically
    return label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help="Path to your test image")
    args = parser.parse_args()
    label = main(args.image)
    print(label)