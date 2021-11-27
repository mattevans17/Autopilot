import numpy as np
import cv2

FRAME_SIZE = (640, 480)
VIDEO_URL = 'media/video.mp4'
BLUR_MATRIX = (1, 1)

CLASSIFIER_POINTS_FREQ = 40
CLASSIFIER_MIN_LANE_WIDTH = 3
CLASSIFIER_OFFSET_X_MAX = 50
CLASSIFIER_CLASS_MAX_SIZE = 20


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


def lane_points(original):
    classes = []
    frame = np.copy(original)
    frame[frame == 255] = 1
    shape = np.array(frame).shape
    offset = shape[0] // CLASSIFIER_POINTS_FREQ
    points = []
    for idx in range(shape[0]):
        if idx % offset == 0:
            row = frame[idx]
            matches_num = 0

            for idx_1 in range(shape[1]):
                curr_value = row[idx_1]
                if curr_value == 1:
                    matches_num += 1

                elif matches_num >= CLASSIFIER_MIN_LANE_WIDTH:
                    point = (idx_1, idx)

                    if idx == 0:
                        classes.append([point])

                    else:
                        current_include_in_class = False
                        for class_idx in range(len(classes)):
                            try:
                                prev_pt_x = classes[class_idx][(idx // offset) - 1][0]

                            except:
                                prev_pt_x = classes[class_idx][len(classes[class_idx])-1][0]

                            if abs(point[0] - prev_pt_x) < CLASSIFIER_OFFSET_X_MAX:
                                classes[class_idx].append(point)
                                current_include_in_class = True
                                break

                        if not current_include_in_class:
                            classes.append([point])

                    matches_num = 0

    classes = [_class for _class in classes if len(_class) > CLASSIFIER_CLASS_MAX_SIZE]
    return classes


def main():
    cap = cv2.VideoCapture(VIDEO_URL)

    while True:
        ret, frame = cap.read(cv2.IMREAD_GRAYSCALE)
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            break

        resized = resize_frame(frame)
        blurred = gauassian_blur(resized)
        yw_filtered = yw_filter(blurred)
        top_view = get_top_view(yw_filtered)
        original_top_view = get_top_view(resized)

        pts_classes = lane_points(top_view)
        mask = np.zeros(original_top_view.shape, dtype=np.uint8)
        for class_idx in range(len(pts_classes)):
            for point in pts_classes[class_idx]:
                color = (0, 0, 255)
                if class_idx == 0:
                    color = (0, 255, 0)
                elif class_idx == 1:
                    color = (255, 0, 0)
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
