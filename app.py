import sys
import os
sys.path.append(os.path.abspath("src"))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Set environment variables
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Client application class
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

# Route for home page
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

# Route to train the model
@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")  # Triggers your training pipeline
    return "Training done successfully!"

# Route to predict image
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)

# Entry point
if __name__ == "__main__":
    clApp = ClientApp()
    # Changed port to 5001 and host to localhost to avoid socket permission errors
    app.run(host='127.0.0.1', port=5001)
