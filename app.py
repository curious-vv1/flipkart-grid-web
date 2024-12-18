import os
import cv2
import numpy as np
from flask import Flask, request, render_template, Response, send_file
from main import main

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')  # Renders Product Info


@app.route('/generate-embeddings')
def generate_embeddings():
    # Renders Generate Embeddings
    return render_template('generate_embeddings.html')


@app.route('/fruits-vegetables')
def fruits_vegetables():
    # Renders Fruits & Vegetables
    return render_template('fruits_vegetables.html')


@app.route('/data')
def data():
    return render_template('data.html')


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


@app.route('/upload_fruits_vegetables', methods=['POST'])
def upload_fruits_vegetables():
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


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image file uploaded", 400

    image = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)
    return {"filename": image.filename}, 200


@app.route('/image_feed/<filename>')
def image_feed(filename):
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(image_path):
        return "Image not found", 404
    return send_file(image_path, mimetype='image/jpeg')



if __name__ == "__main__":
    app.run(debug=True)
