from keras.models import load_model
import cv2
import numpy as np
from time import time
import os
import pandas as pd
import random
import csv
import glob
from sklearn import model_selection

from keras.utils.np_utils import to_categorical

FILENAME = './real_tl_classifier.h5'
ROOT_PATH = './'
BATCH_SIZE = 4
IMAGE_HEIGHT = 600
IMAGE_WIDTH = 800
IMAGE_CHANNEL = 3


def get_saved_model(path):
    model = load_model(path)
    return model

# fuction to read image from file


def get_image(index, data):
    # Read image and appropiately traffic light color
    image = cv2.imread(os.path.join(
        ROOT_PATH, data['path'].values[index].strip()))
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    return image


def generator(data):
    while True:
        # Randomize the indices to make an array
        indices_arr = np.random.permutation(data.count()[0])
        for batch in range(0, len(indices_arr), BATCH_SIZE):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + BATCH_SIZE)]

            # initializing the arrays, x_test
            x_test = np.empty(
                [0, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], dtype=np.float32)

            for i in current_batch:
                # get an image and its corresponding color for an traffic light
                image = get_image(i, data)

                # Appending them to existing batch
                x_test = np.append(x_test, [image], axis=0)

            yield x_test


def evaluate_model(data, class_idx):

    test_gen = generator(data)

    loaded_model = get_saved_model(FILENAME)
    scores = loaded_model.predict_generator(
        test_gen,
        steps=len(data)//BATCH_SIZE,
        verbose=1)
    ctr = 0
    for score in scores:
        if np.argmax(score) == class_idx:
            ctr += 1
    return ctr/len(scores)


if __name__ == "__main__":
    print('Accuracy for Red Light..')
    data = pd.read_csv(os.path.join('./traffic_light_test.csv'))
    data = data[data.apply(lambda x: x['color'] == 'red', axis=1)]
    result = evaluate_model(data, 0)
    print(result*100, '%')

    print('Accuracy for Green/Yellow Light..')
    data = pd.read_csv(os.path.join('./traffic_light_test.csv'))
    data = data[data.apply(lambda x: x['color'] == 'not_red', axis=1)]
    result = evaluate_model(data, 1)
    print(result*100, '%')

    print('Accuracy for images where there is no traffic light..')
    data = pd.read_csv(os.path.join('./traffic_light_test.csv'))
    data = data[data.apply(lambda x: x['color'] == 'not_light', axis=1)]
    result = evaluate_model(data, 2)
    print(result*100, '%')
