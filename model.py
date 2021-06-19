# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:25:21 2021

@author: Samie
"""
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import decode_predictions
import cv2
import os
import numpy as np


model = VGG16(weights='imagenet')

# load an image from file

image = load_img('/content/asign0.jpg', target_size=(224, 224))

# convert the image pixels to a numpy array

image = img_to_array(image)

# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# predict the probability across all output classes
pred = model.predict(image)


# Opening the Video file
def videocapture(location):
    capture= cv2.VideoCapture(location)
    i=0
    while(capture.isOpened()):
        ret, frame = capture.read()
        if ret == False:
            break
   
    image = cv2.imwrite("C:/Users/Samie/Desktop/Untitled Folder/object-detection-vgg16/frames/assign"+str(i)+'.jpg',frame)
    image = cv2.imread("C:/Users/Samie/Desktop/Untitled Folder/object-detection-vgg16/frames/*"+str(i)+'.jpg')
    image = img_to_array(frame)
    image = np.expand_dims(image, axis=0)
    print('%s (%.2f%%)' % (label[1], label[2]*100))
    i+=1
 
    capture.release()
    cv2.destroyAllWindows() 


def detecting(frames):
    location = []
    object = []
    for file in os.listdir('frames/'):
        print(file)
        full_path ='frames/' + file
        image = load_img(full_path, target_size=(224,224))
        image = img_to_array(image)
        image = image.reshape((1,image.shape[0], image.shape[1], image.shape[2]))

        image = preprocess_input(image)
        pred = model.predict(image)
        label = decode_predictions(pred, top = 1)
        object.append(label[0][0][1])
        location.append(full_path)
        print(label)
        print()

    capture.release()
    cv2.destroyAllWindows()


def search(list, items):
    for i in range(len(list)):
        if list[i] == items:
          return location[i]

a = search(list= object, items = 'laptop')
print(a)


