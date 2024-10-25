

from flask import Flask, render_template, request, redirect, url_for, Response
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# modèle YOLO
model = YOLO('/Users/mohammedtbahriti/Documents/TRAFIC_DETECTION/runs/detect/train/weights/best (1).pt')

# sauvegarder les fichiers téléchargés
UPLOAD_FOLDER = '/Users/mohammedtbahriti/Documents/python/yolo_app/uploads'
RESULTS_FOLDER = '/Users/mohammedtbahriti/Documents/python/yolo_app/runs/detect'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    uploaded_file = request.files['file']
    
    if uploaded_file.filename == '':
        return redirect(request.url)

    # Sauvegarder le fichier téléchargé
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
    uploaded_file.save(file_path)

    # Effectuer la détection
    results = model.predict(source=file_path, save=True)

    # Renvoie le nom de la vidéo ou photo avec les résultats
    output_file_name = os.path.basename(file_path)  # Nom du fichier d'origine
    return redirect(url_for('show_results', filename=output_file_name))

@app.route('/results/<filename>')
def show_results(filename):
    # Le fichier de résultats
    result_file_path = os.path.join(RESULTS_FOLDER, filename)
    return render_template('results.html', filename=result_file_path)

@app.route('/video_feed')
def video_feed():
    # Initialiser la capture vidéo à partir de la webcam
    cap = cv2.VideoCapture(0)  

    def generate_frames():
        while True:
            success, frame = cap.read()  # Lire une frame
            if not success:
                break
            else:
                # Encodez la frame en JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                
                # pour le streaming
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
