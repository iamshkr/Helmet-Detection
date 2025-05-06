from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import math
import cvzone
import numpy as np
import base64
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

model = YOLO("Weights/best.pt")
class_labels = ['With Helmet', 'Without Helmet']

@app.route('/detect', methods=['POST'])
def detect_helmet():
    file = request.files['image']
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            if conf > 0.1:
                cvzone.putTextRect(img, f'{class_labels[cls]} {conf}', (x1, y1 - 10),
                                   scale=0.8, thickness=1, colorR=(255, 0, 0))

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
