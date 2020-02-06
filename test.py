from typing import List, Tuple

import cv2
import numpy as np

from hand_tracking.hand_tracker import HandTracker
from hand_tracking.recognizer import IfGestureRecognizer, XgbGestureRecognizer


class GestureModel:
    def __init__(self, recongizer_type, palm_model_path, landmark_model_path, anchors_path):

        self.hand_tracker = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                                        box_shift=0.2, box_enlarge=1.2)
        if recongizer_type == 'xgboost':
            self.recognizer = XgbGestureRecognizer(xgb_path="models/xgb.pkl")
            print("Use xgboost type recognizer")
        else:
            self.recognizer = IfGestureRecognizer()
            print("Use IF type recongizer")

    def forward(self, image: np.ndarray) -> List[Tuple[str, float]]:
        image = self.preprocess_image(image)
        kps, boxes = self.hand_tracker(image)
        if len(kps) == 0:
            return [("_alt", 1.0)]
        gestures = []
        for kp in kps:
            gesture, score = self.recognizer(kp)
            gestures.append((gesture, float(score)))
        return gestures

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


if __name__ == '__main__':
    model = GestureModel(recongizer_type="if",
                         palm_model_path="models/palm_detection_without_custom_op.tflite",
                         landmark_model_path="models/hand_landmark.tflite",
                         anchors_path="data/anchors.csv")

    image = cv2.imread("data/two_hands.jpg")
    result = model.forward(image)
    print(result)
