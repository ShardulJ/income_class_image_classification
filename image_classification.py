import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import tensorflow as tf
import cv2
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as pyplot
from Model.model import Model
# import torch
from sklearn.model_selection import train_test_split 

# reading the excel file
df = pd.read_excel('./TrainingSet_8VWz3PL.xlsx')

image_height = image_width = 128

# number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# number of epochs
epochs = 20

# learning rate
lr = 1e-3

# size of batch
batch_size = 64

# number of classes
numLabels = 5

# size of image when flattened to a single dimension
img_size_flat = image_height * image_width * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (image_height, image_width)

# how long to wait after validation loss stops improving before terminating training
early_stopping = None                     


# function to load all the images 
def load_data(folder,dataFrame, col):
    images = []
    for filename in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.resize(img, (image_height,image_width))
        img = img.astype("float32") / 255.0
        if img is not None:
            images.append(img)
    X = np.array(images)

    y_val = df[col].values
    y = to_categorical(y_val)

    return X, y

X, y = load_images_from_folder('./Training_Images', dataFrame = df, col = 'label')

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

print(X_train.shape)
print(X_test.shape)

# account for skew in the labeled data
classTotals = y_train.sum(axis=0)
classWeight = classTotals.max() / classTotals

# augment the dataset to increase the number of training samples
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

# Initializing the optimizer and the model
print("compiling model")
opt = Adam(lr=lr, decay=lr / (epochs * 0.5))
model = Model.build(width=image_width, height=image_height, depth=3,
    classes=numLabels)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# compile and train
print("training network")
H = model.fit__generator(
    aug.flow(X_train, y_train, batch_size=batch_size),
    validation_split=0.2,
    steps_per_epoch=X_train.shape[0] // batch_size,
    epochs=epochs,
    class_weight=classWeight,
    verbose=1)
    
# evaluate the network
print("evaluating network")
predictions = model.predict(X_test, batch_size=batch_size)
print(accuracy_score(y_test.argmax(axis=1),predictions.argmax(axis=1)))
print(classification_report(y_test.argmax(axis=1),predictions.argmax(axis=1))

print("serializing the model")
model.save('./output')

# plot the training loss and accuracy
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('./output')















