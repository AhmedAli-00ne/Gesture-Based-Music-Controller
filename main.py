from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import tensorflow as tf
from tensorflow import keras
from tensorflow import image as tf_image
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import spotipy
import cv2
import keyboard
import time
from keras.models import load_model
import joblib

cap = cv2.VideoCapture(0)
##########################Time Control##############################################
debounce_time = 5.0
last_gesture = None
last_execution_time = time.time()
##########################Spotify##############################################
client_id = os.environ.get("SPOTIPY_CLIENT_ID")
client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
redirect_uri = os.environ.get("SPOTIPY_REDIRECT_URI", "http://localhost:5000/callback")

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                client_secret=client_secret,
                                                redirect_uri=redirect_uri,
                                                scope="user-modify-playback-state user-read-playback-state"))

playlist_uri = "https://open.spotify.com/playlist/Token key"

devices = sp.devices()

device_id = devices['devices'][0]['id']

gesture_mapping = {
    '': sp.start_playback,
    '05_thumb': sp.pause_playback,
    '09_c': sp.next_track,
    '': sp.previous_track,
    '01_palm': sp.volume,
    '10_down': sp.volume
}
##########################CNN/KNN Model##############################################
lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('leapgestrecog/leapGestRecog/00/'):
    lookup[j] = count
    reverselookup[count] = j
    count = count + 1
loaded_model = load_model('gesture_recognition_model.h5')
#loaded_model = joblib.load('knn_model.joblib')

def recognize_gesture_CNN(image):
    img = Image.fromarray(image).convert('L')
    img = img.resize((320, 120))
    arr = np.array(img)
    t_test = arr.reshape((1, 120, 320, 1))
    t_test = t_test / 255.0
    predictions = loaded_model.predict(t_test)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0, predicted_class]
    if confidence > 0.86:
        predicted_gesture = reverselookup[predicted_class]
        return predicted_gesture
    else:
        return None


def recognize_gesture_KNN(image):
    img = Image.fromarray(image).convert('L')
    img = img.resize((320, 120))
    img_array = np.array(img) / 255.0 
    flattened_img = img_array.flatten() 
    flattened_imd = flattened_img.reshape(1,-1)
    prediction = loaded_model.predict(flattened_imd)
    predicted_gesture = reverselookup[prediction[0]]
    return predicted_gesture


while True:
    ret, frame = cap.read()
    predicted_gesture = recognize_gesture_CNN(frame.copy())
    print(predicted_gesture)
    current_time = time.time()
    if (current_time - last_execution_time) > debounce_time:
        if predicted_gesture in gesture_mapping:
            try:
                if predicted_gesture == '10_down':
                    current_volume = devices['devices'][0]['volume_percent']
                    new_volume = max(current_volume - 18, 0)
                    gesture_mapping[predicted_gesture](volume_percent=new_volume, device_id=device_id)
                elif predicted_gesture == '01_palm':
                    current_volume = devices['devices'][0]['volume_percent']
                    new_volume = min(current_volume + 18, 100)
                    gesture_mapping[predicted_gesture](volume_percent=new_volume, device_id=device_id)
                else:
                    gesture_mapping[predicted_gesture](device_id=device_id)
                last_gesture = predicted_gesture
                last_execution_time = current_time
            except:
                pass
    cv2.imshow("Gesture", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
