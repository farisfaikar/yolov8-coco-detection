from ultralytics import YOLO
import cv2

# Load yolov8 model
model = YOLO("yolov8n.pt")

# Web cam capture
cap = cv2.VideoCapture(0)

ret = True

# Read frames
while ret:
  ret, frame = cap.read()

  # Detect objects

  # Track objects
  results = model.track(frame, persist=True)

  # Plot results
  frame_ = results[0].plot()

  # Visualize
  cv2.imshow("frame", frame_)

  if cv2.waitKey(25) & 0xFF == ord("q"):
    break
