# Descrption

With this project it is scripted how the dataset and table can
be connected in BigQuery. Using the dataset, the data preprocessing 
and feature engineering has been done, after which a model with neural networks
has been trained to solve a regression problem for predicting house prices.
The prediction cen be implemented on the local host connected
with Rest API. Also, 2 Dockerfiles have been created for containerizing both the
training part of the model and the prediction part of the created model.

### Notice
It is worth mentioning, that the trained model has very low train accuracy (approximately 37%),
even though features have been selected based on their correlation with the price and
the model has been trained with tuned hyperparameters. Generally, neural networks
do not perform quite good when it comes to regression problems. To improve the model's accuracy 
a deeper neural network can be trained or simply linear regression with additional kernels
would perform better.


###About

In the folder 'assets' 2 csv files can be found:
- data_table.csv - the dataset is the csv form of the house_prices table connected from BigQuery
- data_test_n.csv - the dataset is left to test the Rest API /predict host. The file can be uploaded and after running 
the export file will automatically be downloaded which will include already predicted prices as a column in the dataset.

In the 'model' folder the trained model is saved 'price_prediction_model.h5' and 'config.ini' file which includes the 
price column's mean and standard deviation
which is used to denormalize predicted prices after prediction.

In the folder 'resources' credentials for accessing BigQuery are located as 'credential_view.json'. The credentials are removed intentionally,
please move the credentials to that directory

- script.py - the file includes the scripts for accessing BigQuery datasets, preprocessing data, building, training and saving the 
price prediction model with neural networks
- training.py - the file is created for executing the script.py
- app.py - the script includes connection with Rest API and prediction with a random dataset (here data_test_n.csv can be used)
- helpers.py - the file includes intermediate functions which are used in script.py

2 Dockerfiles have been created as containers, one of them is a container for training script (Dockerfile_training), the other
one is a container for app.py.
Dockerfile_training is a container for creating and training the prediction model.
Dockerfile is a container for Rest Application Server.





## Start
In the credential_view.json file please paste your key access to BigQuery datasets.

After accessing the house_price table from BigQuery, the csv file should be
copied to assets/data_table.csv.

To start the application please run :
```
docker build -t sentium:latest -f Dockerfile
docker run -p3000:5000 sentium:latest
```
The application will be available with this address: `localhost:3000`

The pretrained model is already available on the directory model/ , but you
can re-execute the tool by running:

```
docker build -t training:latest -f Dockerfile_training
docker run training
```
The model will be created in directory model/

## Endpoint

The predict endpoint is accessible with this path: `/predict`



