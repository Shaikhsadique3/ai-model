from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/generate-report', methods=['POST'])
def generate_report():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        # Here we will add the logic for processing the CSV, running the model, and generating the report
        # For now, let's just return a success message
        return jsonify({"status": "success", "report_url": "/reports/dummy_report.pdf"})

if __name__ == '__main__':
    app.run(debug=True)