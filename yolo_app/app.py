from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from ultralytics import YOLO
import glob

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'runs/detect'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = YOLO('../runs/detect/train/weights/best.pt')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
def get_predicted_image_path(original_filename):
    predict_dir = os.path.join(RESULTS_FOLDER, 'predict')
    pattern = os.path.join(predict_dir, f"{os.path.splitext(original_filename)[0]}*")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)
    
    results = model.predict(source=input_path, save=True)
    
    return redirect(url_for('results', filename=file.filename))

@app.route('/results/<filename>')
def results(filename):
    predicted_path = get_predicted_image_path(filename)
    if predicted_path is None:
        return "Image non trouvée", 404
    
    return render_template('results.html', 
                         image=f'/results/image/{filename}',
                         download_url=f'/download/{filename}')

@app.route('/results/image/<filename>')
def serve_image(filename):
    predicted_path = get_predicted_image_path(filename)
    if predicted_path is None:
        return "Image non trouvée", 404
    return send_file(predicted_path)

@app.route('/download/<filename>')
def download(filename):
    predicted_path = get_predicted_image_path(filename)
    if predicted_path is None:
        return "Image non trouvée", 404
    return send_file(predicted_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)