from ultralytics import YOLO
import cv2

# Load the custom-trained YOLOv8 model
model = YOLO('.\\best.pt')

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the current frame
    results = model(frame)

    # Display the results
    annotated_frame = results[0].plot()  # Plot detections on the frame
    cv2.imshow('YOLOv8 Gesture Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
