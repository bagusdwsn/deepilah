from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.resnet import preprocess_input
import os
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)

# muat model
model = load_model('model/garbage_deploy.h5')

# nama kelas
classes = ["Baterai", "Biologis", "Kaca Cokelat", "Kardus", "Kain", "Kaca Hijau", "Besi", "Kertas", "Plastik", "Sepatu", "Sampah", "Kaca"]

# folder upload
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# atur ekstensi file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser may submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Perform classification
            result = classify_image(filepath)
            return render_template('result.html', result=result, filepath=filepath)
    return render_template('index.html')
@app.route('/clear_uploads')
def clear_uploads():
    # Remove all files in the upload folder
    files = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
    for f in files:
        os.remove(f)
    return redirect(url_for('upload_file'))

def classify_image(filepath):
    image = load_img(filepath, target_size=(224, 224))  
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class
if __name__ == '__main__':
    app.run(debug=True)
