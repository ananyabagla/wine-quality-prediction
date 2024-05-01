from flask import Flask, request, jsonify , render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model_path = 'model2_knn.pkl'
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise FileNotFoundError("Model file not found.")
except Exception as e:
    raise RuntimeError("Error loading the model: {}".format(e))

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        features = [float(request.form.get(f)) for f in request.form]

        # Predict
        prediction = model.predict([features])

        # Return prediction as JSON
        return jsonify(prediction=int(prediction[0]))
    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
