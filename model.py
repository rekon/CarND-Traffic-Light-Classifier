import os
import numpy as np
import pandas as pd
import random
import cv2
import csv
import glob
from sklearn import model_selection

from keras import backend as K
from keras import models, optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, Activation, MaxPooling2D, Reshape
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

ROOT_PATH = './'
BATCH_SIZE = 4
EPOCHS = 20

IMAGE_HEIGHT = 1096
IMAGE_WIDTH = 1368
IMAGE_CHANNEL = 3

TOP_CROP = 266
BOTTOM_CROP = 326

# check for GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

if not os.path.exists('./traffic_light.csv'):
    with open('traffic_light.csv', 'w', newline='') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(['path', 'class', 'color'])

        for myclass, directory in enumerate(['nolight', 'red', 'yellow', 'green']):
            for filename in glob.glob('./data/real_training_data/{}/*.jpg'.format(directory)):
                filename = '/'.join(filename.split('\\'))
                mywriter.writerow([filename, myclass, directory])
    print('CSV file created successfully')
else:
    print('CSV already present')


def random_brightness(image):
    # Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Generate new random brightness
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand*hsv[:, :, 2]
    # Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def zoom(image):
    zoom_pix = random.randint(0, 10)
    zoom_factor = 1 + (2*zoom_pix)/IMAGE_HEIGHT
    image = cv2.resize(image, None, fx=zoom_factor,
                       fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    top_crop = (image.shape[0] - IMAGE_HEIGHT)//2
#     bottom_crop = image.shape[0] - top_crop - IMAGE_HEIGHT
    left_crop = (image.shape[1] - IMAGE_WIDTH)//2
#     right_crop = image.shape[1] - left_crop - IMAGE_WIDTH
    image = image[top_crop: top_crop+IMAGE_HEIGHT,
                  left_crop: left_crop+IMAGE_WIDTH]
    return image


# fuction to read image from file
def get_image(index, data, should_augment):
    # Read image and appropiately traffic light color
    image = cv2.imread(os.path.join(
        ROOT_PATH, data['path'].values[index].strip()))
    color = data['class'].values[index]
    lucky = random.randint(0, 1)
    too_lucky = random.randint(0, 1)
    unlucky = random.randint(0, 1)
    if should_augment:
        if lucky == 1:
            image = random_brightness(image)
        if too_lucky == 1:
            image = cv2.flip(image, 1)
        if not unlucky == 1:
            image = zoom(image)

    return [image, color]

# generator function to return images batchwise


def generator(data, should_augment=False):
    while True:
        # Randomize the indices to make an array
        indices_arr = np.random.permutation(data.count()[0])
        for batch in range(0, len(indices_arr), BATCH_SIZE):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + BATCH_SIZE)]

            # initializing the arrays, x_train and y_train
            x_train = np.empty(
                [0, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], dtype=np.float32)
            y_train = np.empty([0], dtype=np.int32)

            for i in current_batch:
                # get an image and its corresponding color for an traffic light
                [image, color] = get_image(i, data, should_augment)

                # Appending them to existing batch
                x_train = np.append(x_train, [image], axis=0)
                y_train = np.append(y_train, [color])
            y_train = to_categorical(y_train, num_classes=3)

            yield (x_train, y_train)


def get_model(time_len=1):

    model = Sequential()
    model.add(Cropping2D(cropping=((266, 326), (0, 0)),
                         input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)))
    model.add(Lambda(lambda x: x/127.5 - 1.))
    model.add(Conv2D(32, 8, strides=(4, 4), padding="same", activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, 4, strides=(2, 2), padding="same", activation='relu'))
    model.add(MaxPooling2D(2, 2))
#     model.add(Conv2D(64, 5, strides=(2, 2), padding="same", activation='relu'))
    model.add(Flatten())
    model.add(Dropout(.35))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(4))
    model.add(Lambda(lambda x: K.exp(x) / K.sum(K.exp(x))))

    model.compile(optimizer=Adam(lr=1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


if __name__ == "__main__":

    data = pd.read_csv(os.path.join('./traffic_light.csv'))

    # Split data into random training and validation sets
    d_train, d_valid = model_selection.train_test_split(data, test_size=.2)

    train_gen = generator(d_train, True)
    validation_gen = generator(d_valid, False)

    model = get_model()
    model.fit_generator(
        train_gen,
        steps_per_epoch=len(d_train)//BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_gen,
        validation_steps=len(d_valid)//BATCH_SIZE,
        verbose=2
    )
    print("Saving model..")

    model.save("./tl_classifier_keras.h5")

    print("Model Saved successfully!!")

    # Destroying the current TF graph to avoid clutter from old models / layers
    K.clear_session()
