import os
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model
model = load_model('BrainTumor10Epochsclassifier.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Function to map prediction to a class label
def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

# Function to preprocess the image and make the prediction
def getResult(img_path):
    # Read the image using OpenCV
    image = cv2.imread(img_path)
    # Convert the image to RGB and resize to 64x64
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    # Add a batch dimension for model input
    input_img = np.expand_dims(image, axis=0)
    # Predict the class (No Tumor or Tumor)
    result = np.argmax(model.predict(input_img), axis=1)
    return result

# Route for the main page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for the About page
@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

# Route for Login (placeholder for now)
@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to the 'static/uploads' folder
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'static', 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        # Perform prediction
        value = getResult(file_path)
        result = get_className(value[0])

        # Return the result and filename to the webpage
        return render_template('index.html', result=result, filename=secure_filename(f.filename))
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
