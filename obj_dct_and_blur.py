'''object detection and blurring'''
import cv2
from ultralytics import YOLO
import numpy as np
from scipy.spatial.distance import cdist

class ObjectDetector:
    def __init__(self, model_path, target_class, confidence_threshold=0.5, distance_threshold=300):
        """
        Constructor to initialize the model and set parameters.

        Args:
            model_path (str): Path to the YOLO model file.
            target_class (str): Target class name for detection.
            confidence_threshold (float): Confidence threshold for detection.
            distance_threshold (int): Distance threshold for clustering bounding boxes.
        """
        self.model = YOLO(model_path)
        self.target_class = target_class
        self.confidence_threshold = confidence_threshold
        self.distance_threshold = distance_threshold

    def cluster_boxes(self, boxes):
        """
        Cluster bounding boxes based on distance and merge them into a single box.

        Args:
            boxes (list): List of bounding boxes in the format [x1, y1, x2, y2].

        Returns:
            list: Merged bounding boxes.
        """
        if not boxes:
            return []

        # Calculate the centers of the bounding boxes
        centers = np.array([[((box[0] + box[2]) // 2), ((box[1] + box[3]) // 2)] for box in boxes])
        dist_matrix = cdist(centers, centers)  # Compute pairwise distances

        clusters = []
        while len(centers) > 0:
            cluster = []
            indices = np.where(dist_matrix[0] <= self.distance_threshold)[0]
            for idx in indices:
                cluster.append(boxes[idx])
            clusters.append(cluster)

            # Remove elements in the cluster
            dist_matrix = np.delete(dist_matrix, indices, axis=0)
            dist_matrix = np.delete(dist_matrix, indices, axis=1)
            centers = np.delete(centers, indices, axis=0)
            boxes = [box for i, box in enumerate(boxes) if i not in indices]

        # Merge bounding boxes within each cluster
        merged_boxes = []
        for cluster in clusters:
            x1 = min([box[0] for box in cluster])
            y1 = min([box[1] for box in cluster])
            x2 = max([box[2] for box in cluster])
            y2 = max([box[3] for box in cluster])
            merged_boxes.append([x1, y1, x2, y2])

        return merged_boxes

    def detect_and_blur(self, frame):
        """
        Detect the target object in the frame and apply a blur effect.

        Args:
            frame (numpy.ndarray): Input video frame.

        Returns:
            numpy.ndarray: Processed frame with blur applied and target areas restored.
        """
        # Convert the frame to RGB for YOLO model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(rgb_frame, conf=self.confidence_threshold)

        # Collect bounding boxes for the target class
        target_boxes = []
        for result in results[0].boxes:
            cls = int(result.cls)
            conf = result.conf
            if self.model.names[cls] == self.target_class and conf > self.confidence_threshold:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                target_boxes.append([x1, y1, x2, y2])

        # Merge bounding boxes based on proximity
        merged_boxes = self.cluster_boxes(target_boxes)

        # Apply a strong blur effect to the entire frame
        blurred_frame = cv2.GaussianBlur(frame, (51, 51), 0)

        # Restore the original areas inside the bounding boxes and draw rectangles
        for box in merged_boxes:
            x1, y1, x2, y2 = box
            blurred_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
            cv2.rectangle(blurred_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return blurred_frame

def main():
    ########## parameters ######################################
    model_path = 'model/yolo11n.pt'
    target_class = 'toothbrush'
    confidence_threshold = 0.1
    distance_threshold = 300
    ############################################################

    detector = ObjectDetector(model_path, target_class, confidence_threshold, distance_threshold)

    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Failed to open the camera.")
        return

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture the frame.")
            break

        # Process the frame and apply detection and blurring
        processed_frame = detector.detect_and_blur(frame)
        cv2.imshow('Object Detection with Blur', processed_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()