from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the trained model
model_path = './models/Resnet pest.h5'
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading the model:", e)

# Define a function to preprocess the uploaded image
def preprocess_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values
        return img_array
    except Exception as e:
        print("Error preprocessing image:", e)
        return None

# Define a function to make predictions
def make_prediction(image_path):
    try:
        img_array = preprocess_image(image_path)
        if img_array is not None:
            prediction = model.predict(img_array)
            return prediction
        else:
            return None
    except Exception as e:
        print("Error making prediction:", e)
        return None

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/get-pesticide')
def get_pesticide():
    return render_template('get-pesticide.html')

@app.route('/detect-pest')
def detect_pest():
    return render_template('detect-pest.html')

@app.route('/detect-pest', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file:
                file_path = './images/' + file.filename  # Change this path
                file.save(file_path)
                prediction = make_prediction(file_path)
                if prediction is not None:
                    print("Prediction:", prediction)
                    # Convert prediction to human-readable format if needed
                    return render_template('detect-pest.html', prediction=prediction)
                else:
                    return render_template('detect-pest.html', prediction="Error processing image.")
        except Exception as e:
            print("Error uploading file:", e)
            return render_template('detect-pest.html', prediction="Error uploading file.")
    return render_template('detect-pest.html')

if __name__ == '__main__':
    app.run(debug=True)
