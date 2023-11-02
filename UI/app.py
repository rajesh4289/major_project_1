from flask import Flask, render_template, request
import torch
from PIL import Image
from torchvision import transforms
import models

app = Flask(__name)

# Initialize the model and transform
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.model(pretrained=False, requires_grad=False).to(device)
model.load_state_dict(torch.load('path_to_your_model.pth', map_location=device))
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No file part")
        image = request.files['image']
        if image.filename == '':
            return render_template('index.html', error="No selected file")

        input_image = Image.open(image).convert('RGB')
        input_image = transform(input_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_image)
            outputs = torch.sigmoid(outputs)
            sorted_indices = torch.argsort(outputs[0], descending=True)
            best = sorted_indices[:3]

        predicted_genres = [genres[i] for i in best]

        return render_template('index.html', predicted_genres=predicted_genres)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
