
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:49:54 2021

@author: Samie
"""
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import pickle
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import decode_predictions
import cv2
import os
import numpy as np


app = Flask(__name__)
model = VGG16()
       

@app.route("/", methods =['GET'])
def home():
	return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    videofile = request.files['videofile']
    video_path = "./uploads/" + videofile.filename
    videofile.save(video_path)
    
    while(videofile.isOpened()):
        ret, frame = videofile.read()
        if ret == False:
            break
   
    image = cv2.imwrite("./frames/assign"+str(i)+'.jpg',frame)
    image = cv2.imread("./frames/"+str(i)+'.jpg')
    image = img_to_array(frame)
    image = np.expand_dims(image, axis=0)
    classification = '%s (%.2f%%)' % (label[1], label[2]*100)
    i+=1

    location = []
    object = []
    for file in os.listdir('./uploads/'):
        print(file)
        full_path ='./uploads/' + file
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

    return render_template("index.html", prediction = classification)

if __name__=='__main__':
    app.run(host='localhost', port=3000, debug=True)
    #app.run(, debug=True)
	