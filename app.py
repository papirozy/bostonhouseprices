import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

# create flask app
app = Flask(__name__)  #starting point of application

# Load pickle file (Model)
with open('myPickle', 'rb') as f:
    regModel = pickle.load(f)
with open('myScalerPickle', 'rb') as f:
    scaler = pickle.load(f)
    
# Create routes
@app.route('/')
def home(): # defines home page
    return render_template('home.html')


# create a predict API
@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    # data will be gotten as key:value (cos it's json)
    #convert data to a list, then an array, before reshaping
    print(np.array(list(data.values())).reshape(1,-1))
    # produce our transformed data, ready for prediction
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    # output will come as a 2-DIM array, se we are picking the 1st value
    output = regModel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

# in order to run the above
if __name__=="__main__":
    app.run(debug = True)

    
  
    
    