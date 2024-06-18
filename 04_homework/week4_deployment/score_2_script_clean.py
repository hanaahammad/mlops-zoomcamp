import pickle
import os
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import sys


import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("green-taxi-duration")


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    df['ride_id'] = generate_uuids(len(df))
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


def prepare_dictionaries(df: pd.DataFrame):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


# In[ ]:


def load_model(run_id):

    logged_model = f'mlflow-artifacts:/7/{run_id}/artifacts/model'
    print(logged_model)
    model = mlflow.pyfunc.load_model(logged_model)
    
    return model


# In[ ]:


def apply_model(input_file, run_id,output_file ):
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)
    model=load_model(run_id)
    
    y_pred=model.predict(dicts) 
    df_results=pd.DataFrame()
    df_results['tpep_pickup_datetime']=df['tpep_pickup_datetime']
    df_results['PULocationID']=df['PULocationID']
    df_results['DOLocationID']=df['DOLocationID']

    df_results['actual_duration']=df['duration']
    df_results['predicted_duration']=y_pred
    df_results['diff']=df_results['actual_duration']-df_results['predicted_duration']

    df_results['model_version']=run_id
    df_results.to_parquet(output_file, index=False)
    return df_results
    





# In[ ]:
def run():
    #taxi_type = sys.argv[1] #green
    year =2021 # int(sys.argv[1]) # 2021
    month = 3 #int(sys.argv[2]) # 2021

    input_file = "/home/hhammad/Desktop/DTE_MLOps/week4_deployment/yellow_tripdata_2023-03.parquet"
    print(input_file)

    run_id = "e31de181bf78483ba5813fc01d0f7e98"
    print(run_id)

    year = 2023
    month = 3
    # input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    output_file = f'output/yellow-{year:04d}-{month:02d}.parquet'
    print(output_file)
    apply_model(input_file=input_file, run_id = run_id, output_file=output_file)


if __name__ == "__main__":
    run()
