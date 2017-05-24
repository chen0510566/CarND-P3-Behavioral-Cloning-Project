import csv
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
from random import random

import scipy.misc


def load_data_from_csv(csv_file_paths):
    images = {'center': [], 'left': [], 'right': []}
    steer = {'center': [], 'left': [], 'right': []}
    for csv_file_path in csv_file_paths:
        with open(csv_file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                steer_center = float(row[3])
                # create adjust steering measurements for the side camera images
                correction = 0.225
                steer_left = np.min(steer_center + correction, 1.0)
                steer_right = np.max(steer_center - correction, -1.0)
                steer['center'].append(steer_center)
                steer['left'].append(steer_left)
                steer['right'].append(steer_right)

                # read images from center, left and right cameras
                image_center = np.asarray(Image.open(row[0]))
                image_left = np.asarray(Image.open(row[1]))
                image_right = np.asarray(Image.open(row[2]))

                images['center'].append(image_center)
                images['left'].append(image_left)
                images['right'].append(image_right)
    images['center'] = np.asarray(images['center'])
    images['left'] = np.asarray(images['left'])
    images['right'] = np.asarray(images['right'])
    steer['center'] = np.asarray(steer['center'])
    steer['left'] = np.asarray(steer['left'])
    steer['right'] = np.asarray(steer['right'])
    return images, steer


def load_steer_angles(csv_file_paths):
    steer = {'center': [], 'left': [], 'right': [], 'flipped': []}
    for csv_file_path in csv_file_paths:
        with open(csv_file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                steer_center = float(row[3])
                # if abs(steer_center) > 0.001 or random() < 0.02:
                # create adjust steering measurements for the side camera images
                correction = 0.225
                steer_left = min(steer_center + correction, 1.0)
                steer_right = max(steer_center - correction, -1.0)
                steer['center'].append(steer_center)
                steer['left'].append(steer_left)
                steer['right'].append(steer_right)
                steer['flipped'].append(-steer_center)

    steer['center'] = np.asarray(steer['center'])
    steer['left'] = np.asarray(steer['left'])
    steer['right'] = np.asarray(steer['right'])
    return steer


def load_raw_steer_angles(csv_file_paths):
    steer = {'center': [], 'left': [], 'right': []}
    for csv_file_path in csv_file_paths:
        with open(csv_file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                steer_center = float(row[3])
                # if abs(steer_center)>0.001 or random()<0.02:
                # create adjust steering measurements for the side camera images
                correction = 0.225
                steer_left = min(steer_center + correction, 1.0)
                steer_right = max(steer_center - correction, -1.0)
                steer['center'].append(steer_center)
                steer['left'].append(steer_left)
                steer['right'].append(steer_right)
    steer['center'] = np.asarray(steer['center'])
    steer['left'] = np.asarray(steer['left'])
    steer['right'] = np.asarray(steer['right'])
    return steer


def load_csv_records(csv_file_paths):
    samples = []
    for csv_file_path in csv_file_paths:
        with open(csv_file_path) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                # raw images and steer angles
                samples.append([row[0], float(row[3]), 0])
                samples.append([row[1], float(row[3]) + 0.229, 0])
                samples.append([row[2], float(row[3]) - 0.229, 0])
                # raw images and flipped angles;
                samples.append([row[0], -float(row[3]), 1])
                samples.append([row[1], -(float(row[3]) + 0.229), 1])
                samples.append([row[2], -(float(row[3]) - 0.229), 1])

    shuffle(samples)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            steer = []
            for sample in batch_samples:
                original_img = scipy.misc.imread(sample[0])
                cropped_img = original_img[50:140, :]
                # resized_img = scipy.misc.imresize(cropped_img, (90, 320))

                if int(sample[2]) == 0:  # not filpped
                    images.append(cropped_img)
                    steer.append(float(sample[1]))
                else:  # flipped
                    fliped_img = np.fliplr(cropped_img)
                    images.append(fliped_img)
                    steer.append(float(sample[1]))

            X_train = np.array(images)
            y_train = np.array(steer)
            shuffle(X_train, y_train)
            yield X_train, y_train
