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


# predict a number from the mnist numbers
imgnum = random.randint(0,10000)
im2arr = train_data[imgnum]
im2arr = im2arr.reshape(1,784)
y_pred = model.predict_classes(im2arr)
print(y_pred[0])

im2arr = im2arr.reshape(28,28)
plt.figure()
plt.imshow(im2arr, cmap="gray")
#im2arr.save('output.jpg') 
#plt.title("Label: "+str(train_labels[idx]))
plt.title("Predicted: "+str(y_pred[0]))
plt.show()


# predict a diferent number
img = load_img('3.png', grayscale=True)

img = np.resize(img, (28,28,1))
im2arr = img.reshape(1,784)
# invert colors if image is white background and black number
im2arr = np.invert(im2arr)
y_pred = model.predict_classes(im2arr)
print(y_pred[0])

im2arr = im2arr.reshape(28,28)
plt.figure()
plt.imshow(im2arr, cmap="gray")
#im2arr.save('output.jpg') 
#plt.title("Label: "+str(train_labels[idx]))
plt.title("Predicted: "+str(y_pred[0]))
plt.show()