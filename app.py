
#Importing 'Flask' package from 'flask' library
from flask import Flask, render_template, request, jsonify
import pickle
#import pandas as pd
import numpy as np

app = Flask(__name__)
ml_model = pickle.load(open('KnnModel.pkl','rb'))


@app.route('/')
def home():
    return render_template('dp.html')


@app.route("/predict", methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    model_prediction = ml_model.predict(final)
    
    output = model_prediction[0]
    
    return render_template('prediction.html',prediction_text = 'RFM SCORE = {}'.format(output))
   


if __name__ == "__main__":
    app.run(debug=True)


