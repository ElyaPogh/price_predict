# from werkzeug.utils import secure_filename
from flask import Flask, flash, request, make_response
import helpers
from script import Preprocess, Modeling
import pandas as pd
import numpy as np
import configparser

app = Flask(__name__)
app.secret_key = '123456'
mod = Modeling()
loaded_model = mod.load_model('model/price_prediction_model.h5')


@app.route('/')
def home():

    return "Welcome ! :)"

@app.route('/predict' , methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
        file = request.files['file']
        if file:
            df = pd.read_csv(file).iloc[: , 1:]
            df = df.loc[:, df.columns != 'price']
            df_obj = Preprocess(df)
            x = df_obj.feature_engineering()
            x = helpers.shape_normalization(loaded_model, x)
            x = x.fillna(0)
            x = np.asarray(x).astype('float32')
            preds = loaded_model.predict(x)
            config = configparser.ConfigParser()
            config.read('model/config.ini')
            df_std = int(config['main']['df_std'])
            df_mean = int(config['main']['df_mean'])
            preds_list = [int(abs(y*df_std) + abs(df_mean)) for y in preds]
            df['price'] = preds_list
            resp = make_response(df.to_csv())
            resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
            resp.headers["Content-Type"] = "text/csv"
            return resp

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
