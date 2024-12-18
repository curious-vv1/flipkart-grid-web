import cv2
from ultralytics import YOLO
import numpy as np
import gradio as gr
import base64
from io import BytesIO
from PIL import Image

model = YOLO("best.pt")
resize_factor = 3

X1, Y1 = 25, 500
X2, Y2 = 2125, 2000
roi_coordinates = (X1, Y1, X2, Y2)

def model_predict(img, model, width, height, resize_factor, X1, Y1, X2, Y2):
    frame = img.copy()
    cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 0, 0), 20)  
    roi_frame = frame[Y1:Y2, X1:X2]
    results = model.predict(source=roi_frame, verbose=False)

    if hasattr(results[0], 'masks') and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  
        combined_mask = np.any(masks, axis=0).astype(np.uint8)  
        affected_frame = (combined_mask * 255).astype(np.uint8)  
        area = cv2.countNonZero(affected_frame)
    else:
        area = 0

    segmented_frame = results[0].plot()
    frame[Y1:Y2, X1:X2] = segmented_frame
    resized_frame = cv2.resize(segmented_frame, (width // resize_factor, height // resize_factor))

    return resized_frame, area

def full_area_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    empty = np.zeros_like(image)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 40:
            cv2.drawContours(empty, [contour], -1, (255, 255, 255), -1)
    binary_mask = cv2.cvtColor(empty.copy(), cv2.COLOR_BGR2GRAY)
    area = cv2.countNonZero(binary_mask)

    return empty, area

def process_video_frame(frame):
    width = frame.shape[1]
    height = frame.shape[0]
    X1, Y1, X2, Y2 = roi_coordinates

    predicted_frame, area_rotten = model_predict(frame, model, width, height, resize_factor, X1, Y1, X2, Y2)
    h, w, _ = predicted_frame.shape

    full_seg, full_area = full_area_segmentation(frame)
    full_seg = cv2.resize(full_seg, (w, h))
    rotten_percent = (area_rotten / full_area) * 100 if full_area > 0 else 0

    if rotten_percent >= 4:
        cv2.putText(predicted_frame, f"Freshness Score: 0", (w // 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    elif 2 <= rotten_percent < 4:
        cv2.putText(predicted_frame, f"Freshness Score: 1", (w // 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    elif 0.5 <= rotten_percent < 2:
        cv2.putText(predicted_frame, f"Freshness Score: 2", (w // 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    elif 0.1 <= rotten_percent < 0.5:
        cv2.putText(predicted_frame, f"Freshness Score: 3", (w // 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    elif rotten_percent < 0.1:
        cv2.putText(predicted_frame, f"Freshness Score: 4", (w // 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    collage = np.hstack((predicted_frame, full_seg))
    return collage

def process_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return "Error: Could not open video. Ensure the file is not corrupted."

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_video_frame(frame)
            frames.append(processed_frame)

        cap.release()

        output_video_path = "processed_video.mp4"
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

        for frame in frames:
            out.write(frame)
        out.release()

        return output_video_path
    except ValueError as ve:
        return f"ValueError: {str(ve)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


# def gradio_interface(video_file):
#     output_video_path = process_video(video_file)
#     return output_video_path

iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Input Video"),
    outputs=gr.Video(label="Processed Video"),
    title="Live Freshness Detection",
    description="Upload a video to see live frame-by-frame processing.",
    live=True  
)

if __name__ == "__main__":
    iface.launch()
