import cv2
from ultralytics import YOLO
from threading import Event

class FaceRecognition:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.stop_event = Event()
        self.detected_labels = set()  # Store unique detected labels

    def process_frame(self, frame):
        """Run the YOLO model on a single frame and return processed frame."""
        results = self.model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = self.model.names[int(box.cls)]
                self.detected_labels.add(label)  # Save label

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def start_video_stream(self):
        """Start video capture and process frames."""
        cap = cv2.VideoCapture(0)
        self.stop_event.clear()
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.process_frame(frame)
        finally:
            cap.release()

    def stop_video_stream(self):
        """Stop the video stream."""
        self.stop_event.set()
