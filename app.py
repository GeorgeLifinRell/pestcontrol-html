from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import firebase_admin
from firebase_admin import credentials, auth

# Initialize the Firebase Admin SDK
cred = credentials.Certificate('Pesticide_Firebase_Service_Account.json')
firebase_admin.initialize_app(cred)

# Create a new user with email and password
def create_firebase_user(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        print("Successfully created user: {0}".format(user.uid))
        return user.uid
    except Exception as e:
        print("Error creating user: {0}".format(e))
        return None
    
# Check if a user can sign in with email and password
def check_signin_with_email_and_password(email, password):
    try:
        user = auth.get_user_by_email(email=email)
        print("User signed in successfully: {0}".format(user.uid))
        return 'success'
    except auth.UserNotFoundError:
        print("User not found: {0}".format(email))
        return 'user_not_found'
    except Exception as e:
        print("Error checking sign-in: {0}".format(e))
        return 'error'

# Load the trained model
model_path = './models/res.h5'
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

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def handle_login_request():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        login_result = check_signin_with_email_and_password(email=email, password=password)
        if login_result == 'success':
            return home()
        elif login_result == 'wrong_password':
            return jsonify({'error': login_result})
        elif login_result == 'user_not_found':
            return jsonify({'error': login_result})
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/signup', methods=['GET', 'POST'])
def handle_signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user_creation_result = create_firebase_user(email=email, password=password)
        if user_creation_result:
            return login()
    return render_template

@app.route('/get-pesticide')
def get_pesticide():
    return render_template('get-pesticide.html')

@app.route('/detect-pest')
def detect_pest():
    return render_template('detect-pest.html')

@app.route('/detect-pest', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        pest_names = ['aphids', 'armyworm', 'beetle', 'bollworm', 'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem borer']
        try:
            file = request.files['file']
            if file:
                file_path = './images/' + file.filename  # Change this path
                file.save(file_path)
                prediction = make_prediction(file_path)
                if prediction is not None:
                    print("Prediction:", prediction)
                    # Convert prediction to human-readable format if needed
                    # Find the index with the maximum value
                    max_index = np.argmax(prediction)
                    pest_predicted = pest_names[max_index]
                    print(pest_predicted)
                    return jsonify({'pest_predicted': pest_predicted})
                    # return render_template('detect-pest.html', predict=str(pest_predicted))
                else:
                    # return render_template('detect-pest.html', error="Error processing image.")
                    return jsonify({'error': 'Error processing image'})
        except Exception as e:
            print("Error uploading file:", e)
            # return render_template('detect-pest.html', error="Error uploading file.")
            return jsonify({ 'error': 'Error uploading file.' })
    return render_template('detect-pest.html')

if __name__ == '__main__':
    app.run(debug=True)
