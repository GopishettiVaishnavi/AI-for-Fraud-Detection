from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model_bundle = joblib.load("model.joblib")
clf = model_bundle['model']
scaler = model_bundle['scaler']

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    proba = clf.predict_proba(features_scaled)[0,1]
    prediction = int(proba > 0.5)
    return jsonify({"fraud_probability": float(proba), "is_fraud": prediction})

if __name__ == "__main__":
    app.run(debug=True)