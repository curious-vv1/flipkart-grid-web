import cv2
from ultralytics import YOLO
import numpy as np


model = YOLO("best.pt")
video_path = "output2.mp4"

resize_factor = 3

X1, Y1 = 25, 500
X2, Y2 = 2125, 2000

roi_coordinates = (X1, Y1, X2, Y2)


def model_predict(img, model, width, height, resize_factor, X1, Y1, X2, Y2):
    frame = img.copy()
    cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 0, 0), 20)  
    roi_frame = frame[Y1:Y2, X1:X2]
    results = model.predict(source=roi_frame, verbose = False)
    
    if hasattr(results[0], 'masks') and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  
        combined_mask = np.any(masks, axis=0).astype(np.uint8)  

        affected_frame = (combined_mask * 255).astype(np.uint8)  

    area = cv2.countNonZero(affected_frame)

    segmented_frame = results[0].plot()
    frame[Y1:Y2, X1:X2] = segmented_frame
    resized_frame = cv2.resize(segmented_frame, (width // resize_factor, height // resize_factor))

    return resized_frame, area

def kmeans_segmentation(roi_frame, k=2):

    pixel_values = roi_frame.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    segmented_roi = centers[labels.flatten()]
    segmented_roi = segmented_roi.reshape(roi_frame.shape)

    gray_roi = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((9, 9), np.uint8)  
    eroded_mask = cv2.erode(mask, kernel, iterations=2)
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=5)

    return dilated_mask

def area_calculated(frame, width, height, resize_factor, X1, Y1, X2, Y2):
    resized_frame = cv2.resize(frame, (int(width / resize_factor),
                                        int(height / resize_factor)))

    roi_X1 = max(0, int(X1 / resize_factor))
    roi_Y1 = max(0, int(Y1 / resize_factor))
    roi_X2 = min(resized_frame.shape[1], int(X2 / resize_factor))
    roi_Y2 = min(resized_frame.shape[0], int(Y2 / resize_factor))

    roi_frame = resized_frame[roi_Y1:roi_Y2, roi_X1:roi_X2]

    if roi_frame.size == 0:
        print(f"Error: Empty ROI with coordinates {(roi_X1, roi_Y1, roi_X2, roi_Y2)}")
        return resized_frame  
  
    mask = kmeans_segmentation(roi_frame, k=2)
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    resized_frame[roi_Y1:roi_Y2, roi_X1:roi_X2] = mask_3channel

    return resized_frame

def full_area_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

    contours,_ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    empty = np.zeros_like(image)
    for contour in contours:
            area = cv2.contourArea(contour)
            if area>40:
                cv2.drawContours(empty, [contour], -1, (255, 255, 255), -1)
    binary_mask = cv2.cvtColor(empty.copy(), cv2.COLOR_BGR2GRAY)
    area = cv2.countNonZero(binary_mask)

    return empty, area

def main(frame):
    X1, Y1, X2, Y2 = roi_coordinates
    cap = cv2.VideoCapture(video_path)

 
    if not cap.isOpened():
        print("Error: Could not open video.")
        return  


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Press 'q' to exit:")
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Error: Could not read frame.")
    #         break

        # processed_frame = area_calculated(frame, width, height, resize_factor, X1, Y1, X2, Y2)
    predicted_frame, area_rotten = model_predict(frame, model, width, height, resize_factor, X1, Y1, X2, Y2)
    h, w, _ = predicted_frame.shape


    full_seg, full_area = full_area_segmentation(frame)
    full_seg = cv2.resize(full_seg, (w, h))
    rotten_percent = (area_rotten/full_area)*100

    if rotten_percent>=4:
            cv2.putText(predicted_frame, f"Freshness Score:{0}", (w//2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    elif 2<=rotten_percent<4:
            cv2.putText(predicted_frame, f"Freshness Score:{1}", (w//2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    elif 0.5<=rotten_percent<2:
            cv2.putText(predicted_frame, f"Freshness Score:{2}", (w//2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    elif 0.1<=rotten_percent<0.5:
            cv2.putText(predicted_frame, f"Freshness Score:{3}", (w//2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    elif rotten_percent<0.1:
            cv2.putText(predicted_frame, f"Freshness Score:{4}", (w//2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
    collage = np.hstack((predicted_frame, full_seg))

    return predicted_frame
        # _, buffer = cv2.imencode('.jpg', predicted_frame)
        # frame = buffer.tobytes()

        # cv2.imshow('Collage Frame', collage)
 
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # cap.release()
    # cv2.destroyAllWindows()
    # yield (b'--frame\r\n'
    #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    main(video_path, model, resize_factor, roi_coordinates)