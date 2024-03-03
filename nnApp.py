from flask import Flask, jsonify
from flask_cors import CORS
import torch
from triage_neuralnet import NeuralNetwork  # Ensure this import works

app = Flask(__name__)
CORS(app)

# Load the model (ensure the model path is correct and accessible)
model_path = 'path/to/heartattackmodel_complete.pth'
model = torch.load(model_path)
model.eval()

@app.route('/test-prediction', methods=['GET'])
def test_prediction():
    # Predefined input array
    input_tensor = torch.tensor([[1,1,0,1,0,0,1,0,1,1]], dtype=torch.float32)  # Ensure shape matches model's expectation
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()  # Get the predicted class

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
