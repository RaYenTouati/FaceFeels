from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import pygame
from skimage.feature import hog
from sklearn.svm import SVC
# train_test_split n'est plus nécessaire ici car les données sont déjà splittées
from sklearn.preprocessing import StandardScaler
import shutil
from werkzeug.utils import secure_filename
from flask_cors import CORS
import time
import logging
from collections import defaultdict
import warnings
from sklearn.exceptions import ConvergenceWarning
import joblib # Pour sauvegarder/charger le modèle

# --- 0. Configuration Initiale ---
# Suppression des warnings indésirables (laisser pour le développement, retirer en prod si gênant)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app) # Active CORS pour toutes les routes, à affiner si nécessaire en production

# --- 1. Constantes et Chemins ---
UPLOAD_FOLDER = 'static/uploads'
MUSIC_FOLDER = 'music'
# Chemins des nouveaux datasets pour l'entraînement et le test
TRAIN_DATASET_PATH = 'C:\\Users\\ALPHA\\Desktop\\Data\\Train'
TEST_DATASET_PATH = 'C:\\Users\\ALPHA\\Desktop\\Data\\Test'
MODEL_PATH = 'model/svm_model.joblib' # Chemin pour sauvegarder/charger le modèle
SCALER_PATH = 'model/scaler.joblib'   # Chemin pour sauvegarder/charger le scaler
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MUSIC_FOLDER'] = MUSIC_FOLDER
app.config['TRAIN_DATASET_PATH'] = TRAIN_DATASET_PATH
app.config['TEST_DATASET_PATH'] = TEST_DATASET_PATH
app.config['MODEL_PATH'] = MODEL_PATH
app.config['SCALER_PATH'] = SCALER_PATH

# Création des dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MUSIC_FOLDER'], exist_ok=True)
os.makedirs(os.path.dirname(app.config['MODEL_PATH']), exist_ok=True) # Créer le dossier 'model'

# --- 2. Configuration du Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(), # Affiche les logs dans la console
        logging.FileHandler('app.log') # Sauvegarde les logs dans un fichier
    ]
)
logger = logging.getLogger(__name__)

# --- 3. Dictionnaire des Musiques (chemin absolu préférable si l'application est déployée différemment) ---
EMOTION_SONGS = {
    "angry": os.path.join(MUSIC_FOLDER, "angry song.mp3"),
    "happy": os.path.join(MUSIC_FOLDER, "Happy song.mp3"),
    "sad": os.path.join(MUSIC_FOLDER, "sad song.mp3"),
    "neutral": os.path.join(MUSIC_FOLDER, "neutral song.mp3"),
    "surprise": os.path.join(MUSIC_FOLDER, "surprise song.mp3"),
    "fear": os.path.join(MUSIC_FOLDER, "fearful song.mp3"),
    "disgust": os.path.join(MUSIC_FOLDER, "disgusted song.mp3")
}

# --- 4. Initialisation du Modèle et du Scaler ---
svm_clf = None
scaler = None
pygame.mixer.init() # Initialiser une seule fois

# --- 5. Fonctions Utilitaires ---

def allowed_file(filename):
    """Vérifie si l'extension du fichier est autorisée."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def list_images(basePath):
    """Générateur qui liste tous les chemins d'images dans un répertoire."""
    image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    for rootDir, _, filenames in os.walk(basePath):
        for filename in filenames:
            if filename.lower().endswith(image_types):
                yield os.path.join(rootDir, filename)

def load_and_preprocess_image(image_path, size=(64, 64)):
    """Charge une image, la convertit en niveaux de gris et la redimensionne."""
    try:
        # cv2.IMREAD_GRAYSCALE lit directement en niveaux de gris, un peu plus rapide
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA) # INTER_AREA est bon pour la réduction
        return resized
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
    """Extrait les caractéristiques HOG d'une image."""
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

def save_model(model, scaler_obj):
    """Sauvegarde le modèle SVM et le scaler."""
    try:
        joblib.dump(model, app.config['MODEL_PATH'])
        joblib.dump(scaler_obj, app.config['SCALER_PATH'])
        logger.info(f"Model saved to {app.config['MODEL_PATH']}")
        logger.info(f"Scaler saved to {app.config['SCALER_PATH']}")
    except Exception as e:
        logger.error(f"Failed to save model/scaler: {e}")

def load_model():
    """Charge le modèle SVM et le scaler."""
    global svm_clf, scaler
    try:
        if os.path.exists(app.config['MODEL_PATH']) and os.path.exists(app.config['SCALER_PATH']):
            svm_clf = joblib.load(app.config['MODEL_PATH'])
            scaler = joblib.load(app.config['SCALER_PATH'])
            logger.info("Model and scaler loaded successfully.")
            return True
        else:
            logger.warning("No pre-trained model or scaler found. Model will be trained.")
            return False
    except Exception as e:
        logger.error(f"Failed to load model/scaler: {e}")
        return False

# --- 6. Fonction d'Entraînement du Modèle ---
def train_model():
    """Entraîne le modèle SVM sur le dataset d'entraînement et l'évalue sur le dataset de test."""
    global svm_clf, scaler

    logger.info("Starting model training process...")
    start_time = time.time()

    # --- Chargement et prétraitement des données d'entraînement ---
    logger.info(f"Loading training images from: {app.config['TRAIN_DATASET_PATH']}")
    train_image_paths = list(list_images(app.config['TRAIN_DATASET_PATH']))

    if not train_image_paths:
        logger.error("No images found in the training dataset directory. Please check TRAIN_DATASET_PATH.")
        raise ValueError("No images found in the training dataset directory")

    X_train = []
    y_train = []
    for i, image_path in enumerate(train_image_paths):
        try:
            processed_image = load_and_preprocess_image(image_path)
            if processed_image is None:
                continue
            hog_features = extract_hog_features(processed_image)
            if hog_features is not None:
                X_train.append(hog_features)
                y_train.append(os.path.basename(os.path.dirname(image_path)))
            if (i + 1) % 500 == 0: # Log toutes les 500 images
                logger.info(f"Processed {i + 1}/{len(train_image_paths)} training images for feature extraction")
        except Exception as e:
            logger.error(f"Error processing {image_path} during training feature extraction: {e}")
    
    if not X_train:
        raise ValueError("No valid features extracted for training. Check training image files and paths.")

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # --- Chargement et prétraitement des données de test ---
    logger.info(f"Loading test images from: {app.config['TEST_DATASET_PATH']}")
    test_image_paths = list(list_images(app.config['TEST_DATASET_PATH']))

    if not test_image_paths:
        logger.warning("No images found in the test dataset directory. Evaluation on test set will be skipped.")
        X_test = np.array([])
        y_test = np.array([])
    else:
        X_test = []
        y_test = []
        for i, image_path in enumerate(test_image_paths):
            try:
                processed_image = load_and_preprocess_image(image_path)
                if processed_image is None:
                    continue
                hog_features = extract_hog_features(processed_image)
                if hog_features is not None:
                    X_test.append(hog_features)
                    y_test.append(os.path.basename(os.path.dirname(image_path)))
                if (i + 1) % 500 == 0: # Log toutes les 500 images
                    logger.info(f"Processed {i + 1}/{len(test_image_paths)} test images for feature extraction")
            except Exception as e:
                logger.error(f"Error processing {image_path} during test feature extraction: {e}")
        
        if not X_test:
            logger.warning("No valid features extracted for testing. Evaluation on test set will be skipped.")
            X_test = np.array([])
            y_test = np.array([])
        else:
            X_test = np.array(X_test)
            y_test = np.array(y_test)

    # Normalisation : Le scaler doit être FIT sur les données d'entraînement seulement, puis TRANSFORMÉ sur les deux.
    logger.info("Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test.size > 0: # Appliquer la même transformation au jeu de test si non vide
        X_test = scaler.transform(X_test)

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Entraînement du modèle
    try:
        logger.info("Training SVM classifier...")
        svm_clf = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,
            verbose=False,
            random_state=42,
            max_iter=10000
        )

        train_start = time.time()
        svm_clf.fit(X_train, y_train)
        logger.info(f"Training completed in {time.time() - train_start:.2f} seconds")

        # Évaluation
        train_score = svm_clf.score(X_train, y_train)
        logger.info(f"Model performance - Train: {train_score:.2%}")

        if X_test.size > 0:
            test_score = svm_clf.score(X_test, y_test)
            logger.info(f"Model performance - Test: {test_score:.2%}")
        else:
            logger.warning("Test set is empty, skipping evaluation on test data.")

        # Sauvegarde du modèle et du scaler après l'entraînement réussi
        save_model(svm_clf, scaler)

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

# --- 7. Fonction de Prédiction (reste inchangée) ---
def predict_emotion(image_path):
    """Prédit l'émotion à partir d'une image."""
    if svm_clf is None or scaler is None:
        logger.error("Model or scaler not initialized for prediction.")
        raise ValueError("Model not trained or scaler not initialized")

    try:
        prediction_start_time = time.time()

        # Prétraitement
        processed_image = load_and_preprocess_image(image_path)
        if processed_image is None:
            raise ValueError("Image processing failed for prediction")

        # Extraction HOG
        hog_features = extract_hog_features(processed_image)
        if hog_features is None:
            raise ValueError("Feature extraction failed for prediction")

        # Normalisation et prédiction
        hog_features = scaler.transform([hog_features])
        prediction = svm_clf.predict(hog_features)[0]
        probabilities = svm_clf.predict_proba(hog_features)[0]

        prediction_time = time.time() - prediction_start_time
        logger.info(f"Prediction for {os.path.basename(image_path)} took {prediction_time:.4f} seconds.")

        return prediction, {cls: float(prob) for cls, prob in zip(svm_clf.classes_, probabilities)}

    except Exception as e:
        logger.error(f"Prediction error for {image_path}: {str(e)}")
        raise

# --- 8. Fonctions de Lecture Audio (restent inchangées) ---
def play_emotion_song(emotion):
    """Charge et joue une chanson d'émotion."""
    song_path = EMOTION_SONGS.get(emotion.lower())
    if song_path and os.path.exists(song_path):
        try:
            if pygame.mixer.music.get_busy(): # Arrêter la musique actuelle si elle est en cours
                pygame.mixer.music.stop()
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            logger.info(f"Playing song: {song_path}")
            return True
        except Exception as e:
            logger.error(f"Error playing song '{song_path}': {str(e)}")
            return False
    logger.warning(f"No song file found for emotion: {emotion} at path: {song_path}")
    return False

# --- 9. Routes Flask (restent inchangées) ---

@app.route('/api/predict', methods=['POST'])
def predict_route():
    """Route pour la prédiction d'émotion à partir d'une image."""
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'file type not allowed'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
        emotion, probabilities = predict_emotion(filepath)

        return jsonify({
            'emotion': emotion,
            'probabilities': probabilities,
            'image_url': f'/static/uploads/{filename}', # Fournir l'URL de l'image si utile
            'success': True
        })

    except ValueError as e:
        logger.error(f"API Prediction ValueError: {str(e)}")
        return jsonify({'error': str(e)}), 400 # 400 Bad Request pour les erreurs de données/modèle non prêt
    except Exception as e:
        logger.error(f"API Prediction general error: {str(e)}")
        return jsonify({'error': 'internal server error'}), 500

    finally:
        # Suppression du fichier après traitement (peut être désactivé pour débogage)
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted uploaded file: {filepath}")

@app.route('/api/play', methods=['POST'])
def play_route():
    """Route pour jouer une chanson basée sur l'émotion."""
    data = request.get_json()
    if not data or 'emotion' not in data:
        return jsonify({'error': 'no emotion provided'}), 400

    emotion = data['emotion'].lower()
    if play_emotion_song(emotion):
        return jsonify({'success': True, 'message': f'playing {emotion} song'})
    else:
        return jsonify({'error': 'song not found or could not be played'}), 404

@app.route('/api/status')
def status_route():
    """Route pour vérifier le statut du modèle et des ressources."""
    return jsonify({
        'model_ready': svm_clf is not None and scaler is not None,
        'emotion_classes': svm_clf.classes_.tolist() if svm_clf is not None else [],
        'songs_available': {k: os.path.exists(v) for k, v in EMOTION_SONGS.items()},
        'train_dataset_path_exists': os.path.exists(app.config['TRAIN_DATASET_PATH']),
        'test_dataset_path_exists': os.path.exists(app.config['TEST_DATASET_PATH'])
    })

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Route pour servir les fichiers uploadés (peut être retirée si les fichiers sont supprimés immédiatement)."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- 10. Initialisation au Démarrage de l'Application ---
if __name__ == '__main__':
    try:
        logger.info("Starting application initialization...")
        # Tenter de charger le modèle, sinon l'entraîner
        if not load_model():
            train_model()
        
        logger.info("Application ready to receive requests.")
        app.run(host='0.0.0.0', port=5000, debug=False) # Désactiver debug en production
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")
        exit(1) # Quitter avec un code d'erreur si l'initialisation échoue
