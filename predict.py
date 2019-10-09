from keras.models import load_model
import os
import cv2
import numpy as np
from PIL import Image
import csv

model = load_model('saved_models\img_orientation5e64b.h5')
print("Model loaded...")

TEST_PATH = 'test'
ROTATED_PATH = 'test_rotated'

PREDICTIONS = {0: 'upright', 1: 'upside_down', 2: 'rotated_right', 3: 'rotated_left'}
# Rotations in degrees needed for each class
ROTATIONS = {0: 0, 1: 180, 2: 90, 3: 270}

X = []

# Reading test set images
for img in os.listdir('test'):
    img = cv2.imread(os.path.join(TEST_PATH, img))
    X.append(np.array(img))

# Predicting classes
print("Making predictions...")
y = model.predict_classes(np.array(X))

corrected = []

# Correcting images rotation and mounting predictions csv file
print("Rotating images...")
with open('test.preds.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
    writer.writerow(['fn', 'label'])
    for idx, img in enumerate(os.listdir('test_rotated')):
        pill_img = Image.open(os.path.join(ROTATED_PATH, img))
        rotated_img = pill_img.rotate(ROTATIONS[y[idx]])
        rotated_img.save(os.path.join(ROTATED_PATH, img))
        corrected.append(np.array(rotated_img))
        writer.writerow([img, PREDICTIONS[y[idx]]])

np.save('corrected_imgs', corrected)