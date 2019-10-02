import pickle
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
import random

# Load the training and train data from the Pickle file
with open("mnist_dataset_unscaled.pickle", "rb") as f:
      train_data, train_labels, train_data, train_labels = pickle.load(f)

model = load_model('best_model.h5')

# cycle through examples and get 10 wrong ones:
count = 0
for x in range(len(train_data)):
    im2arr = train_data[x]
    im2arr = im2arr.reshape(1,784)
    y_pred = model.predict_classes(im2arr)
    if y_pred[0] != train_labels[x]:
        image = train_data[x].reshape(28,28)
        plt.figure()
        plt.imshow(image, cmap="gray_r")
        plt.title("Predicted: "+str(y_pred[0])+" Correct: "+str(train_labels[x]))
        plt.show()
        count = count +1
    if count == 10:
        break