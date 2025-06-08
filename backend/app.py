from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import pygame
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shutil
from werkzeug.utils import secure_filename
from flask_cors import CORS
import time
import logging
from collections import defaultdict
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppression des warnings indésirables
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)
pygame.mixer.init()

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MUSIC_FOLDER = 'music'
DATASET_PATH = 'C:\\Users\\admin\\Desktop\\test'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MUSIC_FOLDER'] = MUSIC_FOLDER
app.config['DATASET_PATH'] = DATASET_PATH

# Création des dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MUSIC_FOLDER'], exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Dictionnaire des musiques
EMOTION_SONGS = {
    "angry": os.path.join(MUSIC_FOLDER, "angry song.mp3"),
    "happy": os.path.join(MUSIC_FOLDER, "Happy song.mp3"),
    "sad": os.path.join(MUSIC_FOLDER, "sad song.mp3"),
    "neutral": os.path.join(MUSIC_FOLDER, "neutral song.mp3"),
    "surprise": os.path.join(MUSIC_FOLDER, "surprise song.mp3"),
    "fear": os.path.join(MUSIC_FOLDER, "fearful song.mp3"),
    "disgust": os.path.join(MUSIC_FOLDER, "disgusted song.mp3")
}

# Initialisation du modèle
svm_clf = None
scaler = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def list_images(basePath):
    image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    for (rootDir, _, filenames) in os.walk(basePath):
        for filename in filenames:
            if filename.lower().endswith(image_types):
                yield os.path.join(rootDir, filename)

def load_and_preprocess_image(image_path, size=64):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or cannot be read: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (size, size))
        return resized
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

def extract_hog_features(image, orientations=7, pixels_per_cell=(8, 8), cells_per_block=(4, 4)):
    try:
        features = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm='L2-Hys',
            feature_vector=True
        )
        return features
    except Exception as e:
        logger.error(f"Error extracting HOG features: {e}")
        return None

def train_model():
    global svm_clf, scaler
    
    logger.info("Starting model training process...")
    start_time = time.time()
    
    # Liste toutes les images disponibles
    image_paths = list(list_images(app.config['DATASET_PATH']))
    
    if not image_paths:
        raise ValueError("No images found in the dataset directory")
    
    # Analyse la structure du dataset
    label_counts = defaultdict(int)
    for path in image_paths:
        label = os.path.basename(os.path.dirname(path))
        label_counts[label] += 1
    
    logger.info(f"Detected emotion classes: {dict(label_counts)}")
    
    # Vérifie qu'il y a au moins 2 classes
    if len(label_counts) < 2:
        error_msg = (f"At least 2 emotion classes required. Found: {list(label_counts.keys())}\n"
                   f"Expected directory structure: {app.config['DATASET_PATH']}/emotion_name/image.jpg")
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Équilibrage des classes (200 images max par classe)
    MAX_IMAGES_PER_CLASS = 200
    balanced_paths = []
    class_counts = defaultdict(int)
    
    for path in image_paths:
        label = os.path.basename(os.path.dirname(path))
        if class_counts[label] < MAX_IMAGES_PER_CLASS:
            balanced_paths.append(path)
            class_counts[label] += 1
    
    logger.info(f"Using {len(balanced_paths)} images (max {MAX_IMAGES_PER_CLASS} per class)")
    
    # Extraction des caractéristiques
    features = []
    labels = []
    
    for i, image_path in enumerate(balanced_paths):
        try:
            processed_image = load_and_preprocess_image(image_path)
            if processed_image is None:
                continue
                
            hog_features = extract_hog_features(processed_image)
            if hog_features is not None:
                features.append(hog_features)
                labels.append(os.path.basename(os.path.dirname(image_path)))
                
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(balanced_paths)} images")
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
    
    if len(features) < 2:
        raise ValueError("Not enough valid images for training (need at least 2)")
    
    logger.info(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
    
    # Conversion en numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Normalisation
    logger.info("Normalizing features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=42, 
        stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Entraînement du modèle
    try:
        logger.info("Training SVM classifier...")
        svm_clf = SVC(
            kernel='linear',
            C=1.0,
            probability=True,
            verbose=False,  # Désactive les messages du SVM
            random_state=42
        )
        
        train_start = time.time()
        svm_clf.fit(X_train, y_train)
        logger.info(f"Training completed in {time.time() - train_start:.2f} seconds")
        
        # Évaluation
        train_score = svm_clf.score(X_train, y_train)
        test_score = svm_clf.score(X_test, y_test)
        logger.info(f"Model performance - Train: {train_score:.2%}, Test: {test_score:.2%}")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

def predict_emotion(image_path):
    if svm_clf is None or scaler is None:
        raise ValueError("Model not trained or scaler not initialized")
    
    try:
        # Prétraitement
        processed_image = load_and_preprocess_image(image_path)
        if processed_image is None:
            raise ValueError("Image processing failed")
        
        # Extraction HOG
        hog_features = extract_hog_features(processed_image)
        if hog_features is None:
            raise ValueError("Feature extraction failed")
        
        # Normalisation et prédiction
        hog_features = scaler.transform([hog_features])
        prediction = svm_clf.predict(hog_features)[0]
        probabilities = svm_clf.predict_proba(hog_features)[0]
        
        return prediction, {cls: float(prob) for cls, prob in zip(svm_clf.classes_, probabilities)}
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise

def play_emotion_song(emotion):
    song_path = EMOTION_SONGS.get(emotion.lower())
    if song_path and os.path.exists(song_path):
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            return True
        except Exception as e:
            logger.error(f"Error playing song: {str(e)}")
            return False
    logger.warning(f"No song found for emotion: {emotion}")
    return False

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(filepath)
        emotion, probabilities = predict_emotion(filepath)
        
        return jsonify({
            'emotion': emotion,
            'probabilities': probabilities,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"API Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/api/play', methods=['POST'])
def play():
    data = request.get_json()
    if not data or 'emotion' not in data:
        return jsonify({'error': 'No emotion provided'}), 400
    
    emotion = data['emotion'].lower()
    if play_emotion_song(emotion):
        return jsonify({'success': True, 'message': f'Playing {emotion} song'})
    else:
        return jsonify({'error': 'Song not found or could not be played'}), 404

@app.route('/api/status')
def status():
    return jsonify({
        'model_ready': svm_clf is not None,
        'emotion_classes': svm_clf.classes_.tolist() if svm_clf else [],
        'songs_available': {k: os.path.exists(v) for k, v in EMOTION_SONGS.items()}
    })

if __name__ == '__main__':
    try:
        logger.info("Starting application initialization...")
        train_model()
        logger.info("Application ready to receive requests")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")
        exit(1)