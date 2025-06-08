import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from skimage.feature import hog
import pygame
import time

# Initialisation de pygame pour jouer la musique
pygame.mixer.init()

# Types d'images supportés
image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_images(basePath, contains=None):
    return list_files(basePath, validExts=image_types, contains=contains)

def list_files(basePath, validExts=None, contains=None):
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue
            ext = filename[filename.rfind("."):].lower()
            if validExts is None or ext.endswith(validExts):
                imagePath = os.path.join(rootDir, filename)
                yield imagePath

def colortogray(im):
    image = cv2.imread(im)
    if image is None:
        raise ValueError(f"Image not found or could not be read: {im}")
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return imgray

def resizeImage(image, size):
    return cv2.resize(image, (size, size))

def feat_lab(imagePaths):
    features = []
    labels = []
    for imagePath in imagePaths:
        im = colortogray(imagePath)
        im = resizeImage(im, 64)
        fd1 = hog(im, orientations=7, pixels_per_cell=(8, 8), cells_per_block=(4, 4), block_norm='L2-Hys', transform_sqrt=False)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
        features.append(fd1)
    features = np.array(features)
    labels = np.array(labels)
    return features, labels

# Charger les images et extraire les caractéristiques
dataset_path = 'C:\\Users\\admin\\Desktop\\test'
imagePaths = list(list_images(dataset_path))
features, labels = feat_lab(imagePaths)

print("[INFO] The number of features ..." + str(features.shape[1]))
print("[INFO] The number of samples ..." + str(labels.shape[0]))

svm_clf = SVC(kernel='rbf', gamma='scale', C=10)

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = classification_report(y_train, pred)
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
    else:
        pred = clf.predict(X_test)
        clf_report = classification_report(y_test, pred)
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

R = random.randint(1, 88)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=40)

print("[INFO] The number of images used in training ..." + str(X_train.shape[0]))
print("[INFO] The number of images used in testing ..." + str(X_test.shape[0]))

svm_clf.fit(X_train, y_train)

print_score(svm_clf, X_train, y_train, X_test, y_test, train=True)
print_score(svm_clf, X_train, y_train, X_test, y_test, train=False)

# Fonction pour prédire l'émotion à partir d'une image donnée
def predict_emotion(imagePath):
    im = colortogray(imagePath)
    im = resizeImage(im, 64)
    fd = hog(im, orientations=7, pixels_per_cell=(8, 8), cells_per_block=(4, 4), block_norm='L2-Hys', transform_sqrt=False)
    fd = np.array([fd])
    pred = svm_clf.predict(fd)
    return pred[0]

# Fonction pour jouer une chanson en fonction de l'émotion prédite
def play_song(emotion):
    songs = {"angry": "C:\\Users\\admin\\Music\\angry song.mp3",
        "disgusted": "C:\\Users\\admin\\Music\\disgusted song.mp3",
        "fearful": "C:\\Users\\admin\\Music\\fearful song.mp3",
        "happy": "C:\\Users\\admin\\Music\\Happy song.mp3", 
        "neutral": "C:\\Users\\admin\\Music\\neutral song.mp3",
        "sad": "C:\\Users\\admin\\Music\\sad song.mp3",
        "surprised": "C:\\Users\\admin\\Music\\surprised song.mp3"
    }
    song_path = songs.get(emotion, None)
    if song_path and os.path.exists(song_path):
        try:
            with open(song_path, 'rb') as f:
                print(f"[DEBUG] Successfully opened file: {song_path}")
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            print(f"[DEBUG] Playing song for emotion: {emotion}")
            while pygame.mixer.music.get_busy():
                print("[DEBUG] Music is playing...")
                time.sleep(1)
        except pygame.error as e:
            print(f"Error playing song for emotion {emotion}: {e}")
    else:
        print(f"No song found for emotion: {emotion}")

# Demander à l'utilisateur de fournir le chemin de l'image
if __name__ == "__main__":
    image_path_to_predict = input("Donne moi le chemin de l'image : ")

    # Vérifier si le chemin fourni existe
    if not os.path.exists(image_path_to_predict):
        print(f"Le chemin spécifié n'existe pas : {image_path_to_predict}")
    else:
        # Prédire l'émotion à partir de l'image
        predicted_emotion = predict_emotion(image_path_to_predict)
        print(f"Predicted Emotion: {predicted_emotion}")

        # Jouer la chanson associée
        play_song(predicted_emotion)
