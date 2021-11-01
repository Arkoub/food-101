import os
import sys
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from util import base64_to_pil


app = Flask(__name__)

model=load_model('models/model.h5')

print('Model loaded. Check http://127.0.0.1:8008/')

def get_classes():
    a_file = open("models/labels.txt", "r")

    list_of_lists = []
    for line in a_file:
      stripped_line = line.strip()
      list_of_lists.append(stripped_line)

    a_file.close()
    print(list_of_lists)
    return list_of_lists

def model_predict(img, model):
    img = img.resize((256, 256))
    x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x, mode='tf')
    #preds = model.predict(x)

    img=tf.keras.applications.xception.preprocess_input(x)
    preds=model.predict(np.array([img]))
   
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        preds = model_predict(img, model)
        print(preds)
        class_names=get_classes()
        result=class_names[np.argmax(preds)]
        pred_proba = "{:.3f}".format(np.amax(preds))   
        
        print(result)

        
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
   app.run(port=8008)
