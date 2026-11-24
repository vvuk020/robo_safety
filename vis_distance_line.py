
import cv2
import numpy as np

from cam_safety_alg_video_optm import SafetyEstimator


def visualize_robot_distance(
    estimator,
    video_path='SavedFilesNP/Video.avi',
    save_video=True,
    output_path='SavedFilesNP/robot_distance_output.avi'
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open input video!")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer
    out = None

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            raise RuntimeError("Cannot create output video!")

    robot_point = (372, 317)

    # Calculate robot 3D position using MiDaS
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Detect humans
        x_pix, y_pix, human_boxes = estimator.yolo.detect_human(frame)

        # Estimate depth
        depth_map = estimator.get_depth(frame)

        # Robot 3D coordinates
        xr, yr, zr = estimator.midas.get_world_coord(
            robot_point[0],
            robot_point[1],
            depth_map
        )

        robot_position = np.array([xr, yr, zr])

        # Draw bounding boxes
        estimator.yolo.draw_boxes(frame, human_boxes)

        # Draw robot
        cv2.circle(frame, robot_point, 8, (0, 0, 255), -1)
        cv2.putText(frame, "ROBOT",
                    (robot_point[0] + 10, robot_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

        # Process each detected human
        for box in human_boxes:

            x1, y1, x2, y2 = map(int, box[:4])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            human_center = (center_x, center_y)

            # Human 3D coordinates
            xh, yh, zh = estimator.midas.get_world_coord(
                center_x,
                center_y,
                depth_map
            )

            human_position = np.array([xh, yh, zh])

            # Calculate 3D distance
            _, distance = estimator.midas.calculate_points_distance(
                human_position,
                robot_position
            )

            # Draw connecting line
            cv2.line(frame, robot_point, human_center,
                     (0, 255, 0), 2)

            # Draw human center
            cv2.circle(frame, human_center, 6, (255, 0, 0), -1)

            # Display distance
            cv2.putText(
                frame,
                f"Distance: {distance:.2f} m",
                (center_x, center_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("Human-Robot Distance", frame)

        # Save processed frame
        if save_video:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    if out is not None:
        out.release()
        print(f"Video saved: {output_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":

    estimator = SafetyEstimator()

    model_path = 'estimator_include/midas_v31_models/weights/dpt_levit_224.pt'
    model_type = 'dpt_levit_224'

    estimator.init_midas_v31(
        model_path=model_path,
        v31_type=model_type
    )

    # Initialize camera intrinsic parameters
    h, w = 780, 1280

    fx = 2.17404891e+03 * 0.5
    fy = 2.20471536e+03 * 0.5
    cx = 3.17695080e+02
    cy = 2.37041115e+02

    intrinsic_param = (h, w, fx, fy, cx, cy)

    estimator.midas.load_intrinsic(intrinsic_param)

    # Run visualization
    visualize_robot_distance(
        estimator,
        save_video=True
    )
