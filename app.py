from flask import Flask, request, jsonify
import pickle
import numpy as np
with open('mlp.pkl', 'rb') as f:
    model = pickle.load(f)

app=Flask(__name__)
@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(port=5000)