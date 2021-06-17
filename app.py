
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:49:54 2021

@author: Samie
"""


from flask import Flask, render_template
import pickle



app = Flask(__name__)
#model = pickle.load(open('modelname', rb))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/")
def index():
    return "Hello this is the new version!"
if __name__=='__main__':
    app.run(host='localhost', debug=True)

