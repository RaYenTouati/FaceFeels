from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import pygame
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import shutil
from werkzeug.utils import secure_filename
from flask_cors import CORS
import time

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)
pygame.mixer.init()

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MUSIC_FOLDER = 'music'
DATASET_PATH = 'C:\\Users\\ALPHA\\Desktop\\Dataset4mo'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MUSIC_FOLDER'] = MUSIC_FOLDER
app.config['DATASET_PATH'] = DATASET_PATH

# Création des dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MUSIC_FOLDER'], exist_ok=True)

# Dictionnaire des musiques
EMOTION_SONGS = {
    "angry": os.path.join(MUSIC_FOLDER, "angry song.mp3"),
    "disgusted": os.path.join(MUSIC_FOLDER, "disgusted song.mp3"),
    "fear": os.path.join(MUSIC_FOLDER, "fearful song.mp3"),
    "happy": os.path.join(MUSIC_FOLDER, "Happy song.mp3"),
    "neutral": os.path.join(MUSIC_FOLDER, "neutral song.mp3"),
    "sad": os.path.join(MUSIC_FOLDER, "sad song.mp3"),
    "surprise": os.path.join(MUSIC_FOLDER, "surprise song.mp3")
}

# Initialisation du modèle
svm_clf = None

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
        print(f"Error processing image: {e}")
        return None

def extract_hog_features(image, orientations=7, pixels_per_cell=(8, 8), cells_per_block=(4, 4)):
    try:
        features = hog(image,
                      orientations=orientations,
                      pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block,
                      block_norm='L2-Hys')
        return features
    except Exception as e:
        print(f"Error extracting HOG features: {e}")
        return None

def train_model():
    global svm_clf
    
    print("Loading dataset and extracting features...")
    start_time = time.time()
    
    image_paths = list(list_images(app.config['DATASET_PATH']))
    features = []
    labels = []
    
    for image_path in image_paths:
        try:
            # Charger et prétraiter l'image
            processed_image = load_and_preprocess_image(image_path)
            if processed_image is None:
                continue
                
            # Extraire les caractéristiques HOG
            hog_features = extract_hog_features(processed_image)
            if hog_features is not None:
                features.append(hog_features)
                # Le label est le nom du dossier parent
                label = os.path.basename(os.path.dirname(image_path))
                labels.append(label)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    if not features:
        raise ValueError("No valid images found in dataset")
    
    print(f"Extracted features from {len(features)} images in {time.time() - start_time:.2f} seconds")
    
    # Convertir en tableaux numpy
    X = np.array(features)
    y = np.array(labels)
    
    # Séparation en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Entraînement du modèle SVM
    print("Training SVM model...")
    svm_clf = SVC(kernel='rbf', gamma='scale', C=10, probability=True)
    svm_clf.fit(X_train, y_train)
    
    # Évaluation
    train_score = svm_clf.score(X_train, y_train)
    test_score = svm_clf.score(X_test, y_test)
    print(f"Training complete - Train accuracy: {train_score:.2f}, Test accuracy: {test_score:.2f}")

def predict_emotion(image_path):
    if svm_clf is None:
        raise ValueError("Model not trained")
    
    # Prétraitement de l'image
    processed_image = load_and_preprocess_image(image_path)
    if processed_image is None:
        raise ValueError("Image processing failed")
    
    # Extraction des caractéristiques HOG
    hog_features = extract_hog_features(processed_image)
    if hog_features is None:
        raise ValueError("Feature extraction failed")
    
    # Prédiction
    prediction = svm_clf.predict([hog_features])[0]
    probabilities = svm_clf.predict_proba([hog_features])[0]
    emotion_labels = svm_clf.classes_
    
    # Création d'un dictionnaire de probabilités
    prob_dict = {label: float(prob) for label, prob in zip(emotion_labels, probabilities)}
    
    return prediction, prob_dict

def play_emotion_song(emotion):
    song_path = EMOTION_SONGS.get(emotion.lower())
    if song_path and os.path.exists(song_path):
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            return True
        except Exception as e:
            print(f"Error playing song: {e}")
            return False
    return False

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            emotion, probabilities = predict_emotion(filepath)
            
            # Supprimer le fichier après traitement
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'emotion': emotion,
                'probabilities': probabilities,
                'success': True
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/play', methods=['POST'])
def play():
    data = request.get_json()
    emotion = data.get('emotion', '').lower()
    
    if not emotion:
        return jsonify({'error': 'No emotion provided'}), 400
    
    if play_emotion_song(emotion):
        return jsonify({'success': True, 'message': f'Playing {emotion} song'})
    else:
        return jsonify({'error': 'Song not found or could not be played'}), 404

@app.route('/api/status')
def status():
    return jsonify({
        'model_loaded': svm_clf is not None,
        'songs_available': {k: os.path.exists(v) for k, v in EMOTION_SONGS.items()}
    })

if __name__ == '__main__':
    # Entraîner le modèle au démarrage
    try:
        train_model()
    except Exception as e:
        print(f"Model training failed: {e}")
        exit(1)
    
    # Démarrer le serveur
    app.run(host='0.0.0.0', port=5000, debug=True)