from flask import Flask, Response, render_template, request
import cv2
from ultralytics import YOLO
import numpy as np
from scipy.spatial.distance import cdist
import threading

class VideoProcessor:
    def __init__(self, model_path, confidence_threshold=0.5, distance_threshold=300):
        """
        Initialize the video processor with YOLO model and thresholds.

        Args:
            model_path (str): Path to the YOLO model.
            confidence_threshold (float): Confidence threshold for detection.
            distance_threshold (int): Distance threshold for clustering bounding boxes.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.distance_threshold = distance_threshold
        self.blur_intensity = 101  # Default blur intensity (must be odd for GaussianBlur)
        self.output_frame = None
        self.lock = threading.Lock()

    def cluster_boxes(self, boxes):
        """
        Cluster bounding boxes based on distance and merge them.

        Args:
            boxes (list): List of bounding boxes in [x1, y1, x2, y2] format.

        Returns:
            list: Merged bounding boxes.
        """
        if not boxes:
            return []

        centers = np.array([[((box[0] + box[2]) // 2), ((box[1] + box[3]) // 2)] for box in boxes])
        dist_matrix = cdist(centers, centers)

        clusters = []
        while len(centers) > 0:
            cluster = []
            indices = np.where(dist_matrix[0] <= self.distance_threshold)[0]
            for idx in indices:
                cluster.append(boxes[idx])
            clusters.append(cluster)
            dist_matrix = np.delete(dist_matrix, indices, axis=0)
            dist_matrix = np.delete(dist_matrix, indices, axis=1)
            centers = np.delete(centers, indices, axis=0)
            boxes = [box for i, box in enumerate(boxes) if i not in indices]

        merged_boxes = []
        for cluster in clusters:
            x1 = min([box[0] for box in cluster])
            y1 = min([box[1] for box in cluster])
            x2 = max([box[2] for box in cluster])
            y2 = max([box[3] for box in cluster])
            merged_boxes.append([x1, y1, x2, y2])

        return merged_boxes

    def process_frame(self, frame):
        """
        Process a single video frame: detect objects and apply blur.

        Args:
            frame (numpy.ndarray): Input video frame.

        Returns:
            numpy.ndarray: Processed video frame.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(rgb_frame, conf=self.confidence_threshold)

        toothbrush_boxes = []
        for result in results[0].boxes:
            cls = int(result.cls)
            conf = result.conf
            if self.model.names[cls] == 'toothbrush' and conf > self.confidence_threshold:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                toothbrush_boxes.append([x1, y1, x2, y2])

        merged_boxes = self.cluster_boxes(toothbrush_boxes)
        blurred_frame = cv2.GaussianBlur(frame, (self.blur_intensity, self.blur_intensity), 0)

        for box in merged_boxes:
            x1, y1, x2, y2 = box
            blurred_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
            cv2.rectangle(blurred_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return blurred_frame

    def capture_video(self):
        """
        Continuously capture video frames, process them, and store the result.
        """
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            with self.lock:
                self.output_frame = processed_frame.copy()

        cap.release()

    def generate_frames(self):
        """
        Yield processed frames in JPEG format for streaming.

        Yields:
            bytes: Encoded JPEG frame.
        """
        while True:
            with self.lock:
                if self.output_frame is None:
                    continue

                flag, encoded_image = cv2.imencode(".jpg", self.output_frame)
                if not flag:
                    continue

            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encoded_image) + b'\r\n')

    def set_blur_intensity(self, intensity):
        """
        Set the blur intensity, ensuring it is an odd number.

        Args:
            intensity (int): Desired blur intensity.
        """
        self.blur_intensity = max(1, intensity // 2 * 2 + 1)

# Initialize Flask app and VideoProcessor
app = Flask(__name__)
video_processor = VideoProcessor(model_path='model/yolo11m.pt')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(video_processor.generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/set_blur", methods=["POST"])
def set_blur():
    try:
        intensity = int(request.form.get("intensity", 51))
        video_processor.set_blur_intensity(intensity)
        return f"Blur intensity set to {video_processor.blur_intensity}", 200
    except ValueError:
        return "Invalid intensity value", 400

if __name__ == "__main__":
    # Start the video capture thread
    threading.Thread(target=video_processor.capture_video, daemon=True).start()

    # Run the Flask app
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True, use_reloader=False)