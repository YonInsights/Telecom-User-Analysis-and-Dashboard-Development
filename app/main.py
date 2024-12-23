from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    # Parse input data
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return response
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
