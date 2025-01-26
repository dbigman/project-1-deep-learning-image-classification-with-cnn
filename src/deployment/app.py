# src/deployment/app.py

from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import io
import os

app = Flask(__name__)

# Load the trained model
model = torch.load('path_to_saved_model.pth')
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    files = request.files.getlist('file')
    predictions = []

    for file in files:
        if file:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img = transform(img).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                outputs = model(img)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            pred_class = class_names[predicted.item()]
            pred_conf = confidence.item()
            predictions.append({
                'filename': file.filename,
                'prediction': pred_class,
                'confidence': round(pred_conf, 4)
            })

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    # To run the app locally, use: python app.py
    app.run(host='0.0.0.0', port=5000, debug=True)
