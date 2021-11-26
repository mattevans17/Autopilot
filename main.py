import numpy as np
import cv2


FRAME_SIZE = (640, 480)
VIDEO_URL = 'media/2.mp4'
BLUR_MATRIX = (1, 1)
POINTS_FREQ = 50
MIN_LANE_WIDTH = 3


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
    frame = np.copy(original)
    frame[frame == 255] = 1
    shape = np.array(frame).shape
    offset = shape[0] // POINTS_FREQ
    points = []
    for idx in range(shape[0]):
        if idx % offset == 0:
            row = frame[idx]
            matches_num = 0
            for idx_1 in range(shape[1]):
                curr_value = row[idx_1]
                if curr_value == 1:
                    matches_num += 1
                elif matches_num >= MIN_LANE_WIDTH:
                    point = (idx_1, idx)
                    points.append(point)
                    matches_num = 0
    return points


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
        points = lane_points(top_view)
        for point in points:
            original_top_view = cv2.circle(original_top_view, point, radius=5, color=(0, 255, 0), thickness=-1)

        cv2.imshow('frame', top_view)
        cv2.imshow('yw_filtered', original_top_view)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
