from ultralytics import YOLO
import cv2


class yolo_det():
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect_human(self, frame):
        results = self.model(frame)
        human_boxes = [
            box.xyxy[0].tolist() for detection in results for box in detection.boxes
            if int(box.cls[0].item()) == 0  # Assuming class_id 0 is 'person'
        ]

        if human_boxes:
            x_min, y_min, x_max, y_max = map(int, human_boxes[0])  # Take the first detected human
            return (x_min + x_max) // 2, (y_min + y_max) // 2, human_boxes

        return None, None, human_boxes

    def draw_boxes(self, frame, boxes):
        for human_box in boxes:
            x_min, y_min, x_max, y_max = map(int, human_box)  # Ensure integer values
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
