import os
import cv2
import numpy as np
from flask import Flask, request, render_template, Response
from main import main

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

    #     # Process the frame here if needed
    #     frame = cv2.putText(frame, "Processing Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = main(frame)

        #     # Encode the frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No video file uploaded", 400

    video = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    return "Video uploaded successfully", 200

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
