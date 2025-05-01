from flask import Flask, render_template, request, session, redirect, url_for
import os
import shutil
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
MAX_IMAGES = 6

# Setup model
model = load_model('model/landmark_model.h5')
with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

key_points = {
    'left_hand': mp_pose.PoseLandmark.LEFT_WRIST,
    'right_hand': mp_pose.PoseLandmark.RIGHT_WRIST,
    'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
    'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
    'nose': mp_pose.PoseLandmark.NOSE,
}

# Prediksi 1 gambar
def predict_letter(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "?"
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return "?"

    landmarks = results.pose_landmarks.landmark
    coords = []

    for idx in key_points.values():
        lm = landmarks[idx]
        coords.extend([lm.x, lm.y])

    left = landmarks[key_points['left_shoulder']]
    right = landmarks[key_points['right_shoulder']]
    coords.extend([(left.x + right.x) / 2, (left.y + right.y) / 2])

    X = np.array(coords).reshape(1, -1)
    prediction = model.predict(X)
    label_index = np.argmax(prediction)
    letter = label_encoder.inverse_transform([label_index])[0]
    return letter

@app.before_request
def create_upload_folder():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'filenames' not in session:
        session['filenames'] = []

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(path)  # Simpan file ke folder yang ditentukan

            filenames = session['filenames']
            filenames.append(filename)
            session['filenames'] = filenames

            return redirect(url_for('index'))

    return render_template('index.html', count=len(session['filenames']), max=MAX_IMAGES, filenames=session['filenames'])

@app.route('/result')
def result():
    filenames = session.get('filenames', [])
    predictions = []

    for fname in filenames:
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        letter = predict_letter(path)
        predictions.append({'filename': fname, 'letter': letter})

    word = ''.join([item['letter'] for item in predictions])
    return render_template('result.html', predictions=predictions, word=word)

@app.route('/reset')
def reset():
    session.clear()
    shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
