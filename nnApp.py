from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from triage_neuralnet import NeuralNetwork  # Ensure this import works

app = Flask(__name__)
# Allow all origins for demo purposes
CORS(app, resources={r"/test-prediction": {"origins": "*", "methods": ["GET", "POST"]}})

@app.route('/test-prediction', methods=['GET', 'POST'])
def test_prediction():
    if request.method == 'GET':
        # Example GET response for testing the endpoint directly in a browser
        return jsonify({'message': 'This is a test prediction endpoint. Please use POST to submit data.'})
    elif request.method == 'POST':
        data = request.get_json()
        condition = data.get('condition')
        input_tensor = torch.tensor([data['answers']], dtype=torch.float32)
        
        model_path = 'NN_Data/heart attackmodel_complete.pth' if condition == 'heartAttack' else 'NN_Data/woundmodel_complete.pth'
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        
        return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
