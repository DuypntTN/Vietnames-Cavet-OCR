from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import json
from server.helpers.clearFolder import clearFolderContent
from main import OcrOfficial


UPLOAD_FOLDER = './run/detect/images'
RETURN_FOLDER = './run/temp/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'json'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def upload_file():
    clearFolderContent(app.config['UPLOAD_FOLDER'])

    def cavet_card():
        # Read the JSON file
        with open('2.json') as file:
            data = json.load(file)
        return jsonify(data)

    if 'file' not in request.files:
        return jsonify({'error': 'input image not provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No input image selected'}), 401
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return cavet_card()
        # return cavet_card(os.path.join(app.config['UPLOAD_FOLDER'], '2.json'))
    else:
        return jsonify({'error': 'Invalid file type'}), 402


@app.route('/cavet', methods=['POST'])
def cavet_card():
    clearFolderContent(app.config['UPLOAD_FOLDER'])
    if 'file' not in request.files:
        return jsonify({'error': 'input image not provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No input image selected'}), 401
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # RUN DETECT HERE
        json = ocr.run()
        # GET THE JSON HERE
        return jsonify(json)
    else:
        return jsonify({'error': 'Invalid file type'}), 402


if __name__ == '__main__':
    ocr = OcrOfficial(
        wc_path="./weights/CavetDetector_v1.pt",
        wcf_path="./weights/CavetFieldsDetecotor_v1.pt"
    )
    ocr.set_image_config(
        im_root_path="./run/detect/images",
        save_cavet_detector_path="./run/temp/",
        save_cavet_fields_detector_path="./run/temp/"
    )
    app.run(debug=True, port=9090)
