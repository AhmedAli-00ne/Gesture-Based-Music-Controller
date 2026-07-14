# Gesture-Based Music Controller

Control Spotify playback with hand gestures. A webcam feed is classified by a
CNN (trained on the [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
dataset), and each gesture is mapped to a Spotify action — play / pause,
next / previous track, volume up / down — through the Spotify Web API (spotipy).

A university computer-vision project · Python · OpenCV · TensorFlow / Keras · scikit-learn.

## How it works

- **`main.py`** — the app: reads the webcam, recognises the gesture with the CNN
  (`gesture_recognition_model.h5`), and drives Spotify playback.
- **`train.py`** — trains the CNN on the LeapGestRecog dataset and saves the model.
- **`trainKnn.py`** — an alternative K-Nearest-Neighbours classifier over the same data.

## Setup

```bash
pip install -r requirements.txt
```

Create an app in the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
and set your credentials as environment variables (spotipy reads these automatically):

```bash
export SPOTIPY_CLIENT_ID=your_client_id
export SPOTIPY_CLIENT_SECRET=your_client_secret
export SPOTIPY_REDIRECT_URI=http://localhost:5000/callback
```

Download the [LeapGestRecog dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
into `leapgestrecog/` (used for training and for the gesture-label map).

## Run

```bash
python main.py     # start the controller (press 'q' to quit)
python train.py    # retrain the CNN
```

## Notes

- The trained CNN (`gesture_recognition_model.h5`) is included. The KNN model
  (`knn_model.joblib`) is not — run `trainKnn.py` to regenerate it.
- Gestures are the 10 LeapGestRecog classes (palm, fist, thumb, index, ok, etc.).
