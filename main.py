from math import sqrt, pow
import numpy as np
import cv2

VIDEO_URL = 'media/video.mp4'
BLUR_MATRIX = (1, 1)

CLASSIFIER_POINTS_FREQ = 40
CLASSIFIER_MIN_LANE_WIDTH = 2
CLASSIFIER_CLASS_MIN_SIZE = 5

LANE_LEFT_POS = 0
LANE_RIGHT_POS = 1
LANE_POINTS_MAX_DIST = 50

FRAME_SIZE = (640, 480)
FRAME_MIDDLE_X = FRAME_SIZE[0] // 2


class Lane:
    def __init__(self):
        self.position = -1
        self.points = []

    def add_point(self, point):
        self.points.append(point)

    def get_last_point(self):
        last_point_idx = len(self.points) - 1
        return self.points[last_point_idx]

    @staticmethod
    def calc_distance(point_1, point_2):
        offset_x = abs(point_1[0] - point_2[0])
        offset_y = abs(point_1[1] - point_2[1])
        return int(sqrt(pow(offset_x, 2) + pow(offset_y, 2)))

    def check_point(self, point):
        if self.calc_distance(self.get_last_point(), point) <= LANE_POINTS_MAX_DIST:
            return True
        return False


class Model:
    def __init__(self):
        self.lanes = []

    def add_lane(self):
        self.lanes.append(Lane())
        return len(self.lanes) - 1


class Classifier:
    def __init__(self, model):
        self.model = model

    def classify(self, point):
        point_included = False
        for lane_idx in range(len(self.model.lanes)):
            curr_lane = self.model.lanes[lane_idx]
            if curr_lane.check_point(point):
                curr_lane.add_point(point)
                point_included = True
                break
        if not point_included:
            lane_idx = self.model.add_lane()
            self.model.lanes[lane_idx].add_point(point)


def gauassian_blur(frame):
    return cv2.GaussianBlur(frame, BLUR_MATRIX, cv2.BORDER_DEFAULT)


def get_top_view(frame):
    pts_1 = np.float32([
        [FRAME_SIZE[0] // 2 - FRAME_SIZE[0] // 6, FRAME_SIZE[1] // 10 * 6],
        [FRAME_SIZE[0] // 2 + FRAME_SIZE[0] // 6, FRAME_SIZE[1] // 10 * 6],
        [0, FRAME_SIZE[1]],
        [FRAME_SIZE[0], FRAME_SIZE[1]]
    ])
    pts_2 = np.float32([
        [0, 0],
        [FRAME_SIZE[0], 0],
        [0, FRAME_SIZE[1] * 2],
        [FRAME_SIZE[0], FRAME_SIZE[1] * 2]
    ])

    matrix = cv2.getPerspectiveTransform(pts_1, pts_2)
    return cv2.warpPerspective(frame, matrix, (FRAME_SIZE[0], FRAME_SIZE[0] * 2))


hsv_min = np.array((0, 0, 175), np.uint8)
hsv_max = np.array((255, 255, 255), np.uint8)


def yw_filter(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    return frame_thresh


def resize_frame(frame):
    return cv2.resize(frame, FRAME_SIZE)


def binary_frame(frame):
    copied = np.copy(frame)
    copied[frame == 255] = 1
    return copied


def classifier_points(frame):
    model = Model()
    shape = np.array(frame).shape
    classifier = Classifier(model)
    offset_y = shape[1] // CLASSIFIER_POINTS_FREQ
    for frame_y in reversed(range(shape[0])):
        if frame_y % offset_y == 0:
            row = frame[frame_y]
            matches_num = 0
            for frame_x in range(shape[1]):
                curr_value = row[frame_x]
                if curr_value == 1:
                    matches_num += 1
                elif matches_num >= CLASSIFIER_MIN_LANE_WIDTH:
                    point = (frame_x, frame_y)
                    classifier.classify(point)
                    matches_num = 0
    return model


def main():
    cap = cv2.VideoCapture(VIDEO_URL)

    while True:
        ret, frame = cap.read(cv2.IMREAD_GRAYSCALE)
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            break

        resized = resize_frame(frame)
        yw_filtered = yw_filter(resized)
        top_view = get_top_view(yw_filtered)
        original_top_view = get_top_view(resized)

        model = classifier_points(binary_frame(top_view))
        mask = np.zeros(original_top_view.shape, dtype=np.uint8)

        for idx in range(len(model.lanes)):
            for point in model.lanes[idx].points:
                color = (0, 0, 255)
                if idx == 0:
                    color = (0, 255, 0)
                elif idx == 1:
                    color = (255, 0, 0)
                elif idx == 2:
                    color = (255, 255, 0)
                elif idx == 3:
                    color = (255, 0, 255)
                elif idx == 4:
                    color = (0, 255, 255)
                elif idx == 5:
                    color = (255, 255, 255)
                mask = cv2.circle(
                    mask, point, radius=5,
                    color=color, thickness=-1
                )
        cv2.imshow('frame', original_top_view)
        cv2.imshow('mask', mask)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
