import numpy as np
import cv2


frame_size = (640, 480)
video_url = 'video.mp4'


def get_perspective(frame):
    pts_1 = np.float32([
        [frame_size[0] // 2 - frame_size[0] // 6, frame_size[1] // 10 * 6],
        [frame_size[0] // 2 + frame_size[0] // 6, frame_size[1] // 10 * 6],
        [0, frame_size[1]],
        [frame_size[0], frame_size[1]]
    ])
    pts_2 = np.float32([
        [0, 0],
        [frame_size[0], 0],
        [0, frame_size[1] * 2],
        [frame_size[0], frame_size[1] * 2]
    ])

    matrix = cv2.getPerspectiveTransform(pts_1, pts_2)
    return cv2.warpPerspective(frame, matrix, (frame_size[0], frame_size[0] * 2))


def main():
    cap = cv2.VideoCapture(video_url)

    while True:
        ret, frame = cap.read(cv2.IMREAD_GRAYSCALE)
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame = cv2.resize(frame, frame_size)
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = get_perspective(result)

        # Binary frame
        thresh = 160
        result = cv2.threshold(result, thresh, 255, cv2.THRESH_BINARY)[1]

        cv2.imshow('frame', frame)
        cv2.imshow('result', result)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
