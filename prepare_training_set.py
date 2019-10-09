import os
import csv
import cv2
import numpy as np

LABELS = {'upright': 0, 'upside_down': 1, 'rotated_right': 2, 'rotated_left': 3}
TRAIN_PATH = 'train'

X = []
y = []

# Getting images and labels from csv file
with open('train.truth.csv') as file:
    csv_reader = csv.reader(file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue
        else:
            img = cv2.imread(os.path.join(TRAIN_PATH, row[0]))
            X.append(np.array(img))
            y.append(np.eye(4)[LABELS[row[1]]])
            line_count += 1

# Saving to file train and test data
np.save('x_train', X[:40000])
np.save('y_train', y[:40000])
np.save('x_test', X[40000:])
np.save('y_test', y[40000:])
