import math

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_distances


def get_euclidean_distance(a_x, a_y, b_x, b_y):
    dist = pow(a_x - b_x, 2) + pow(a_y - b_y, 2)
    return np.sqrt(dist)


class IfGestureRecognizer:
    @staticmethod
    def is_thumb_near_first_finger(point1, point2):
        distance = get_euclidean_distance(point1[0], point1[1], point2[0], point2[1])
        return distance < 0.1  # TODO distance

    def __call__(self, keypoints: np.ndarray) -> (str, float):
        # finger states
        thumb_is_open = False
        first_finger_is_open = False
        second_finger_is_open = False
        third_finger_is_open = False
        fourth_finger_is_open = False

        if keypoints[2][0] > keypoints[0][0]:
            rotation = 'right'
        else:
            rotation = 'left'

        if rotation == 'left':
            if keypoints[3][0] < keypoints[2][0] and keypoints[4][0] < keypoints[2][0]:
                thumb_is_open = True

        if rotation == 'right':
            if keypoints[3][0] > keypoints[2][0] and keypoints[4][0] > keypoints[2][0]:
                thumb_is_open = True

        if keypoints[7][1] < keypoints[5][1] and keypoints[8][1] < keypoints[6][1]:
            first_finger_is_open = True

        if keypoints[11][1] < keypoints[9][1] and keypoints[12][1] < keypoints[10][1]:
            second_finger_is_open = True

        if keypoints[15][1] < keypoints[13][1] and keypoints[16][1] < keypoints[14][1]:
            third_finger_is_open = True

        if keypoints[19][1] < keypoints[17][1] and keypoints[20][1] < keypoints[18][1]:
            fourth_finger_is_open = True

        # Hand gesture recognition
        if thumb_is_open and first_finger_is_open and second_finger_is_open and third_finger_is_open \
                and fourth_finger_is_open:
            gesture = "five"
        elif not thumb_is_open and first_finger_is_open and second_finger_is_open and third_finger_is_open \
                and fourth_finger_is_open:
            gesture = "four"
        elif thumb_is_open and first_finger_is_open and second_finger_is_open and not third_finger_is_open \
                and not fourth_finger_is_open:
            gesture = "three"
        elif thumb_is_open and first_finger_is_open and not second_finger_is_open and not third_finger_is_open \
                and not fourth_finger_is_open:
            gesture = "two"
        elif not thumb_is_open and first_finger_is_open and not second_finger_is_open and not third_finger_is_open \
                and not fourth_finger_is_open:
            gesture = "one"
        elif not thumb_is_open and first_finger_is_open and second_finger_is_open and not third_finger_is_open \
                and not fourth_finger_is_open:
            gesture = "yeah"
        elif not thumb_is_open and first_finger_is_open and not second_finger_is_open and not third_finger_is_open \
                and fourth_finger_is_open:
            gesture = "rock"
        elif thumb_is_open and first_finger_is_open and not second_finger_is_open and not third_finger_is_open \
                and fourth_finger_is_open:
            gesture = "spiderman"
        elif thumb_is_open and not first_finger_is_open and not second_finger_is_open and not third_finger_is_open \
                and fourth_finger_is_open:
            gesture = "six"
        elif not first_finger_is_open and second_finger_is_open and third_finger_is_open and fourth_finger_is_open \
                and self.is_thumb_near_first_finger(keypoints[4], keypoints[8]):
            gesture = "ok"
        else:
            gesture = "_alt"
            print(
                f"Gesture not recognized! Finger States: {thumb_is_open}, {first_finger_is_open}, {second_finger_is_open}, "
                f"{third_finger_is_open}, {fourth_finger_is_open}")
        return gesture, 1.0


class XgbGestureRecognizer:
    def __init__(self, xgb_path):
        self.xgb_model = joblib.load(xgb_path)
        self.iu = np.triu_indices(21, k=1)
        self.mapping = {'_alt': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'yeah': 7,
                        'spiderman': 8, 'rock': 9}
        self.inv_mapping = {v: k for k, v in self.mapping.items()}

    def __call__(self, keypoints: np.ndarray) -> (str, float):
        x0, y0 = keypoints[0]
        delta_x = keypoints[0][0] - keypoints[9][0]
        delta_y = keypoints[0][1] - keypoints[9][1]
        alpha = -math.atan(delta_x / delta_y)

        for j in range(keypoints.shape[0]):
            keypoints[j] = self.to_new_coord_system(keypoints[j][0], keypoints[j][1], x0, y0, alpha)

        dist = cosine_distances(keypoints, keypoints)[self.iu]
        predict = self.xgb_model.predict_proba(np.expand_dims(dist, 0))[0]
        label = np.argmax(predict)
        score = predict[label]
        label = self.inv_mapping[label]
        return label, score

    @staticmethod
    def to_new_coord_system(x, y, x0, y0, alpha):
        new_x = (x - x0) * math.cos(alpha) + (y - y0) * math.sin(alpha)
        new_y = -(x - x0) * math.sin(alpha) + (y - y0) * math.cos(alpha)
        return new_x, -new_y
