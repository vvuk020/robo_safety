
import cv2
import numpy as np

from cam_safety_alg_video_optm import SafetyEstimator


def visualize_robot_distance(estimator, video_path='SavedFilesNP/Video.avi'):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video!")

    robot_point = (372, 317)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Detect humans
        x_pix, y_pix, human_boxes = estimator.yolo.detect_human(frame)

        # Draw bounding boxes
        estimator.yolo.draw_boxes(frame, human_boxes)

        # Draw robot position
        cv2.circle(frame, robot_point, 8, (0, 0, 255), -1)
        cv2.putText(frame, "ROBOT", (robot_point[0] + 10,
                                   robot_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

        # Connect bounding box center to robot
        for box in human_boxes:

            x1, y1, x2, y2 = map(int, box[:4])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            human_center = (center_x, center_y)

            # Draw line
            cv2.line(frame, robot_point, human_center,
                     (0, 255, 0), 2)

            # Draw center of bounding box
            cv2.circle(frame, human_center, 6, (255, 0, 0), -1)

            # Calculate pixel distance
            pixel_distance = np.linalg.norm(
                np.array(robot_point) - np.array(human_center)
            )

            cv2.putText(
                frame,
                f"{pixel_distance:.1f} px",
                (center_x, center_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow("Human-Robot Connection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    estimator = SafetyEstimator()

    visualize_robot_distance(estimator)
