from urllib import response
import requests
from flask import Flask, request, jsonify
import cv2
import tensorflow
import numpy as np
import time
import PIL
from flask_cors import CORS, cross_origin
import os

current = time.time()
app = Flask(__name__)

new_model = tensorflow.keras.models.load_model('model7.h5')
CORS(app)

@app.route('/uploader', methods = ['GET', 'POST'])

def upload_file():
   if request.method == 'POST':
      img = request.files['image']
      
      path=os.path.join(os.getcwd()+ "\\static\\" +img.filename)
      img.save(path)
      
      img = cv2.imread(path)
      new_img = cv2.resize(img,(224,224))

      pred = new_model.predict(np.expand_dims(new_img,axis=0))
      final_pred = np.argmax(pred,axis=1)
      print(final_pred)

      response = None

      if final_pred[0] == 0:
         response = "Normal"
      elif final_pred[0] == 1:
         response = "Cataract"
      elif final_pred[0] == 2:
         response = "Glaucoma"

      return jsonify({'pred':response})

print(time.time()-current)


if __name__ == '__main__':
   app.run(debug = True)