import pickle
import sklearn
print(sklearn.__version__)

import mlflow
from mlflow import MlflowClient


from flask import Flask, request,  jsonify
#MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
RUN_ID = "07c14bc7d78c4b2185738d4a33dba10e"

print(RUN_ID)
logged_model = 'runs:/07c14bc7d78c4b2185738d4a33dba10e/model'
model_uri = f"runs:/{RUN_ID}/model"
#loaded_model = mlflow.pyfunc.load_model(f"runs:/{RUN_ID}/model")
#models:/<model_name>/<model_version>
#RF-regressor, v1
#mlflow-artifacts:/path/to/model
model_name = "RF-regressor"
model_version ='v1'

#loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}@v1")

client = mlflow.tracking.MlflowClient()
experiment_id = "7"
runs = client.search_runs(
    experiment_id,
)
print(runs)
logged_model = 'runs:/07c14bc7d78c4b2185738d4a33dba10e/model'
print(logged_model)
#
art = 'mlflow-artifacts:/http://127.0.0.1:5000/#/experiments/7/runs/07c14bc7d78c4b2185738d4a33dba10e/artifacts'
loaded_model = mlflow.pyfunc.load_model(art)

#f"models:/RF-regressor/v1"
#model = mlflow.sklearn.load_model(model_uri)

#model = mlflow.pyfunc.load_model(logged_model)

#path = client.download_artifacts(run_id=logged_model, path='dict_vectorizer.bin')
#logged_model  = f'runs://{RUN_ID}/model'
#model=mlflow.pyfunc.load_model(logged_model)
#path = client.download_artifacts(run_id=logged_model, path='dict_vectorizer.bin')


def prep_feature(ride):
  print('in prepare feature : ride = ' ,  ride)
  features={}
  features['PU_DO'] = '%s_%s' % (ride['PULocationID'] , ride['DOLocationID'])
  features['trip_distance'] = ride['trip_distance']
  return features

def make_prediction(features):
  #X= dv.transform(features)
  preds= model.predict(features)

  return float(preds)

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict():
  ride = request.get_json()
  print(ride)
  features = prep_feature(ride)
  print(features)
  pred = make_prediction(features)
  print(pred)
  result = {
    'duration': pred[0]
  }
  print(result)
  return jsonify(result)

if __name__ == "__main__":
  app.run(debug=True, host='127.0.0.1', port =9797)
