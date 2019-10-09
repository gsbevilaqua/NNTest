# Neural Network Test

Based on the CIFAR10 model, this Concolutional Neural Network predicts the orientation of 64x64 pixel face photos so that it is possible to correct its orientations.
It reaches 98.33% validation accuracy with 25 epochs. I've tried to make another one a little bit deeper with 1 extra layer in each set of fully connected layers 
and also used the Nadam optimizer instead of RMSProp and in 10 epochs it reaches 98.16% validation accuracy but seems to start overfitting a bit.

## Creating Training Data

Run **prepare_traning_set.py** file with:
```
python prepare_training_set.py
```

From the file **train.truth.csv** it will create 4 numpy array files: **x_train.npy**, **x_test.npy**, **y_train.npy** and **y_test.npy**

## Creating and Training the Model

Run **convNet.py** the same way as before:
```
python convNet.py
```

The model will load and train on the 4 numpy array files. **x_train.npy** contains 40000 images for training and **x_test** holds the 8896 images left 
for validation. The folder **saved_models** must have been created so that the model can be saved, otherwise it will raise an error.

## Correcting the Test Images

To run **predict.py** it is necessary first to create a copy of the test images folder and rename it to **test_rotated**. This is where the images will be rotated.
```
python predict.py (model_name)
```
Example:
```
python predict.py img_orientation25e32b.h5
```
It will also create the file **test.preds.csv** with the predictions the model made for each image on the test folder.
