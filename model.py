import torch
import numpy as np
import cv2
import time
import pyttsx3


def initialize_tts_engine():
    engine = pyttsx3.init()
    engine.say("Welcome to visiotrack.")
    engine.runAndWait()
    return engine


class ObjectDetection:
    def __init__(self, focal_length=None):
        self.model = self.load_model()
        self.classes = self.model.names
        self.focal_length = focal_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.last_announced_time = {}  # Track last announcement time for each object
        self.check_focal_length()
        print("\n\nDevice Used:", self.device)

    def load_model(self):
        # Load the YOLOv5 model from ultralytics/yolov5 repository
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, trust_repo=True)
        return model

    def check_focal_length(self):
        if self.focal_length is None:
            self.focal_length = self.prompt_focal_length()

    @staticmethod
    def prompt_focal_length():
        while True:
            try:
                return float(input("Enter the camera's focal length: "))
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    def score_frame(self, frame):
        # Score the frame using the YOLOv5 model
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        # Map class index to label
        return self.classes[int(x)]

    def estimate_distance(self, width_in_pixels, known_width, focal_length):
        # Estimate distance to the object
        return (known_width * focal_length) / width_in_pixels

    def plot_boxes(self, results, frame):
        labels, cord = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        CONFIDENCE_THRESHOLD = 0.5

        real_width = {"person": 37, "cell phone": 7, "mouse": 6, "bottle": 6.5, "chair": 20}
        detected_objects = []

        for i in range(len(labels)):
            row = cord[i]
            confidence = row[4]
            if confidence >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                object_label = self.class_to_label(labels[i])
                width_in_pixels = x2 - x1
                if object_label in real_width and self.focal_length is not None:
                    known_width = real_width[object_label]
                    distance_cm = self.estimate_distance(width_in_pixels, known_width, self.focal_length)
                    detected_objects.append((distance_cm, x1, y1, x2, y2, object_label, confidence))

        detected_objects.sort(key=lambda x: x[0])  # Sort objects by distance (nearest first)

        # Plot bounding boxes and labels for the three nearest objects
        for obj in detected_objects[:3]:
            distance_cm, x1, y1, x2, y2, object_label, confidence = obj

            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            label_text = f"{object_label} ({confidence:.2f})"
            distance_text = f"Distance: {distance_cm:.2f} cm"

            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
            cv2.putText(frame, distance_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        self.announce_nearest_object(detected_objects)
        return frame

    def announce_nearest_object(self, detected_objects):
        ANNOUNCE_COOLDOWN = 10 
        current_time = time.time()
        for obj in detected_objects:
            distance_cm, _, _, _, _, object_label, _ = obj

            if object_label not in self.last_announced_time or current_time - self.last_announced_time[object_label] > ANNOUNCE_COOLDOWN:
                # Speak the label and distance
                say = f"{object_label}, Distance: {distance_cm:.2f} cm"
                engine.say(say)
                engine.runAndWait()
                self.last_announced_time[object_label] = current_time
                break

    def __call__(self):
        cap = cv2.VideoCapture(0)
        counter = 1

        while cap.isOpened():
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break
            counter += 1
            if counter % 4 == 0:
                # Grayscale Conversion
                # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                results = self.score_frame(frame)
                frame = self.plot_boxes(results, frame)

                end_time = time.perf_counter()
                fps = 1 / np.round(end_time - start_time, 3)
                cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                cv2.imshow("img", frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Initialize text-to-speech engine
engine = initialize_tts_engine()

# Create a new object and execute
detection = ObjectDetection(focal_length=500)  # Pass the focal length as a parameter
detection()
