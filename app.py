import os
import uuid
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename

from models import BloodGroupPrediction
from utils.preprocess import preprocess_image
from utils.feature_extraction import extract_hog_features
from utils.model import load_model, predict_blood_group, save_model, train_svm_model
from utils.dataset import load_dataset, save_sample

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "fingerprintbloodgroupprediction")

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['DATASET_FOLDER'] = 'dataset/dataset_blood_group'
app.config['MODEL_PATH'] = 'models/blood_group_model.pkl'

# Global variable to store the most recent prediction
current_prediction = None

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle fingerprint image upload and prediction"""
    global current_prediction
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
            
        # Check if the file is allowed
        if file and allowed_file(file.filename):
            try:
                # Save the uploaded file
                filename = secure_filename(file.filename)
                unique_filename = f"{str(uuid.uuid4())}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                
                # Preprocess the image
                preprocessed_img = preprocess_image(file_path)
                
                # Extract HOG features
                features = extract_hog_features(preprocessed_img)
                
                # Load model and predict
                try:
                    model_artifact = load_model(app.config['MODEL_PATH'])
                    blood_group, confidence, all_probabilities = predict_blood_group(model_artifact, features)
                    
                    # Create prediction object
                    current_prediction = BloodGroupPrediction(
                        image_path=file_path,
                        blood_group=blood_group,
                        confidence=confidence
                    )
                    
                    # Store prediction in session
                    session['prediction'] = {
                        'image_path': file_path,
                        'blood_group': blood_group,
                        'confidence': float(confidence),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'all_probabilities': [(group, float(prob)) for group, prob in all_probabilities]
                    }
                    
                    # Redirect to results page
                    return redirect(url_for('results'))
                    
                except FileNotFoundError:
                    # Model not found, redirect to model page
                    flash('No trained model found. Please train a model first.', 'error')
                    return redirect(url_for('model'))
                    
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('File type not allowed. Please upload a png, jpg, jpeg, bmp, or tiff file.', 'error')
            return redirect(request.url)
    
    # If GET request or error, render upload form
    return render_template('upload.html')

@app.route('/results')
def results():
    """Display prediction results"""
    # Check if there's a prediction in the session
    if 'prediction' not in session:
        flash('No prediction available. Please upload an image first.', 'error')
        return redirect(url_for('upload'))
        
    prediction = session['prediction']
    
    return render_template('results.html', prediction=prediction)

@app.route('/save-result', methods=['POST'])
def save_result():
    """Save the uploaded image to the dataset with the correct blood group"""
    if 'prediction' not in session:
        flash('No prediction available.', 'error')
        return redirect(url_for('upload'))
        
    if 'blood_group' not in request.form:
        flash('No blood group specified.', 'error')
        return redirect(url_for('results'))
    
    # Get the actual blood group from the form
    actual_blood_group = request.form['blood_group']
    
    # Get the prediction details
    prediction = session['prediction']
    image_path = prediction['image_path']
    predicted_blood_group = prediction['blood_group']
    
    # Save the image to the dataset
    try:
        save_sample(
            image_path=image_path,
            blood_group=actual_blood_group,
            dataset_folder=app.config['DATASET_FOLDER'],
            prediction_blood_group=predicted_blood_group
        )
        
        flash(f'Image saved to the dataset with blood group {actual_blood_group}.', 'success')
        # Clear the current prediction
        session.pop('prediction', None)
        
        return redirect(url_for('dataset'))
        
    except Exception as e:
        flash(f'Error saving image to dataset: {str(e)}', 'error')
        return redirect(url_for('results'))

@app.route('/dataset')
def dataset():
    """Display dataset information"""
    try:
        # Load dataset information
        dataset_info = load_dataset(app.config['DATASET_FOLDER'])
        return render_template('dataset.html', dataset_info=dataset_info)
    except FileNotFoundError:
        flash('Dataset folder not found. Please train a model first.', 'error')
        return render_template('dataset.html', dataset_info=None)
    except Exception as e:
        flash(f'Error loading dataset: {str(e)}', 'error')
        return render_template('dataset.html', dataset_info=None)

@app.route('/diagrams')
def diagrams():
    """Display system diagrams"""
    diagrams_data = [
        {
            'title': 'High Level Architecture',
            'path': '/static/diagrams/high_level_architecture.png',
            'description': 'Overview of the system architecture showing main components and their interactions.'
        },
        {
            'title': 'Class Diagram',
            'path': '/static/diagrams/class_diagram.png',
            'description': 'Shows the classes, their attributes, operations, and relationships in the system.'
        },
        {
            'title': 'Component Diagram',
            'path': '/static/diagrams/component_diagram.png',
            'description': 'Illustrates how components are wired together to form larger components or software systems.'
        },
        {
            'title': 'Sequence Diagram',
            'path': '/static/diagrams/sequence_diagram.png',
            'description': 'Shows object interactions arranged in time sequence, focusing on the exchange of messages.'
        },
        {
            'title': 'Activity Diagram',
            'path': '/static/diagrams/activity_diagram.png',
            'description': 'Depicts the workflow from a start point to the finish point, showing the sequence of activities.'
        },
        {
            'title': 'Deployment Diagram',
            'path': '/static/diagrams/deployment_diagram.png',
            'description': 'Shows the configuration of run-time processing nodes and the components that live on them.'
        }
    ]
    return render_template('diagrams.html', diagrams=diagrams_data)

@app.route('/model', methods=['GET', 'POST'])
def model():
    """Model training and management"""
    model_exists = os.path.exists(app.config['MODEL_PATH'])
    model_info = None
    
    if request.method == 'POST':
        # Handle model training
        action = request.form.get('action')
        
        if action == 'train':
            try:
                # Check if dataset exists
                if not os.path.exists(app.config['DATASET_FOLDER']) or not os.listdir(app.config['DATASET_FOLDER']):
                    flash('Dataset not found. Please manually upload the dataset to the dataset folder.', 'error')
                    return redirect(url_for('model'))
                
                # Load dataset and extract features
                X, y = load_dataset(app.config['DATASET_FOLDER'], return_features=True)
                
                if len(X) == 0 or len(y) == 0:
                    flash('No valid samples found in the dataset.', 'error')
                    return redirect(url_for('model'))
                
                # Train the model
                model_artifact, metrics = train_svm_model(X, y)
                
                # Save the model
                save_model(model_artifact, app.config['MODEL_PATH'])
                
                flash(f'Model trained successfully with accuracy: {metrics["accuracy"]:.2f}', 'success')
                return redirect(url_for('model'))
                
            except Exception as e:
                flash(f'Error training model: {str(e)}', 'error')
                return redirect(url_for('model'))
    
    # For GET requests, display model information if available
    if model_exists:
        try:
            # Load model info
            model_info = {
                'exists': True,
                'path': app.config['MODEL_PATH'],
                'last_modified': datetime.fromtimestamp(
                    os.path.getmtime(app.config['MODEL_PATH'])
                ).strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            flash(f'Error loading model information: {str(e)}', 'error')
    
    return render_template('model.html', model_info=model_info, model_exists=model_exists)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)