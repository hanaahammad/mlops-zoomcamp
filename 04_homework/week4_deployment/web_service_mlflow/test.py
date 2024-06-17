import requests

ride = {
  "PULocationID" :10,
  "DOLocationID" :50,
  "trip_distance" :50
}
url='http://127.0.0.1:9797/predict'
print(ride)
response = requests.post(url, json=ride)
print(response.json())
# features = predict.prepare_feature(ride)
# print(features)
# pred = predict.predict(features)
# print(pred[0])