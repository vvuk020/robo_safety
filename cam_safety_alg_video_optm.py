import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
midas_models_path = os.path.join(current_dir, 'estimator_include')
sys.path.append(midas_models_path)

import matplotlib.pyplot as plt
import cv2
from estimator_include.yoloCls import yolo_det
import time
from estimator_include.midasV30Cls import midasDepth
import numpy as np
from estimator_include.midasV31Cls import midasV31


class SafetyEstimator:
    def __init__(self, model_type="V30"):
        # Midas's initialization and it's output
        self.midas = None
        self.model_type = model_type
        self.raw_midas_output = None
        self.normalized_output = None
        self.midas_true_depth = None

        # Yolo
        self.yolo = None
        self.init_yolo()

        # Calibration points initialization
        self.calibration_points = None
        self.A = None
        self.B = None

        # Arrays for saving
        self.elapsed_time = []
        self.distances_array = []
        self.diff_array = []

    def init_yolo(self):
        self.yolo = yolo_det()
        return self.yolo

    def init_midas_v30(self, model_name=None):
        """
            Default for v30 is MiDaS_small. Other models are:
            "DPT_Large"
            "DPT_Hybrid"
        """
        if model_name is None:
            model_name = "MiDaS_small"
        else:
            model_name = model_name

        self.model_type = "V30"
        self.midas = midasDepth()
        self.midas.init_model(model=model_name)
        self.load_calibration_points()

    def init_midas_v31(self, model_path=None, v31_type=None):
        """
            Default for v31 is dpt_swin2_large_384 model
            :param model_path: 'midas_v31_models/weights/dpt_swin2_large_384.pt'
            :param v31_type: 'dpt_swin2_large_384'
        """
        if model_path is None and v31_type is None:
            model_path = 'estimator_include/midas_v31_models/weights/dpt_swin2_large_384.pt'
            model_v31_type = 'dpt_swin2_large_384'
        else:
            model_path = model_path
            model_v31_type = v31_type

        self.model_type = "V31"
        self.midas = midasV31(
            model_path=model_path,
            model_type=model_v31_type
        )
        self.midas.load_model()
        self.load_calibration_points()

        # Calibration on the sample image is for the initial parameters and is needed for V31 model
        sample_img = cv2.imread('estimator_include/SruvCameraData/3. Frame_RGB_2025-02-22_17-49-24.png')
        self.get_depth(sample_img)

    def load_calibration_points(self, xy_path=None, depth_path=None, isPrint=False):
        """
            Surveillance camera calibration points, default are:
            :param xy_path:    'estimator_include/SruvCameraData/coordinates.json'
            :param depth_path: 'estimator_include/SruvCameraData/distances.json'
        """
        if self.midas is None:
            print("Midas needs to be initialized first!")
            return 1

        if xy_path is None and depth_path is None:
            xy_path = 'estimator_include/SruvCameraData/coordinates.json'
            depth_path = 'estimator_include/SruvCameraData/distances.json'
        else:
            xy_path = xy_path
            depth_path = depth_path

        coordinates = self.midas.read_json(xy_path)
        depth_coordinates = self.midas.read_json(depth_path)
        calibration_points = [(coord['x'], coord['y'], depth['d'])
                              for coord, depth in zip(coordinates, depth_coordinates)]
        self.calibration_points = calibration_points
        if isPrint:
            print("Calibration points are:")
            print(calibration_points)

    def get_depth(self, image):
        """
            Get calibrated estimator_include output.
            Raw and normalized outputs are for debugging purposes.
        """
        if self.model_type == "V30":
            self.raw_midas_output = self.midas.depth_prediction(image)  # V30 prediction
        else:
            self.raw_midas_output = self.midas.process(image)  # V31 prediction

        # Get and calibrate estimator_include from midas
        self.A, self.B, self.normalized_output = self.midas.depth_to_real_lsq(
            midas_prediction=self.raw_midas_output,
            known_points=self.calibration_points,
            isParam=True
        )
        self.midas_true_depth = 1 / (self.A * self.normalized_output + self.B)

        return self.midas_true_depth

    def video_stream(self, path=None, intrinsic_param=None, isPrint=True):
        """
            Method for playing camera/video.
            In method by default are set intrinsic parameters for Surveillance camera
            Intrinsic parameters for laptop camera are:
                Laptop camera
                h, w = 480, 640
                fx, fy = 2.91010521e+03, 2.73899617e+03
                cx, cy = 6.52220285e+02, 3.66729543e+02

            Intrinsic parameters need to be in the form as shown below. (Floats)
            :param path:            'SavedFilesNP/Video.avi'
            :param intrinsic_param: (h, w, fx, fy, cx, cy)
        """
        if self.midas is None or self.yolo is None:
            print("Midas and Yolo need to be initialized first!")
            return 1

        if path is None:
            path = 'SavedFilesNP/Video.avi'
        else:
            path = path

        cap = cv2.VideoCapture(path)

        # Surveillance camera intrinsic parameters
        h, w = 780, 1280
        fx, fy = 2.17404891e+03 * 0.5, 2.20471536e+03 * 0.5
        cx, cy = 3.17695080e+02, 2.37041115e+02
        intrinsic_param_cust = (h, w, fx, fy, cx, cy)

        # Overwrite parameters if passed onto this method
        if intrinsic_param is not None:
            intrinsic_param_cust = intrinsic_param

        self.midas.load_intrinsic(intrinsic_param_cust)

        print("Models initialized.")
        self.process_frames(cap=cap, isPrint=isPrint)

    def process_frames(self, cap, isPrint=False):
        while cap.isOpened():
            start_time = time.time() * 1000
            ret, frame = cap.read()
            height, width = frame.shape[:2]
            if isPrint: print(f"Resolution is: width: {width} x height: {height}")

            if not ret:
                break

            # Get pixels from yolo and estimator_include map from midas
            x_pix, y_pix, human_boxes = self.yolo.detect_human(frame)
            midas_true_depth = self.get_depth(frame)

            # Read world values for a person and the robot
            x, y, z = self.midas.get_world_coord(x_pix, y_pix, midas_true_depth)
            human_position = np.array([x, y, z])
            if isPrint: print(f"Points are x: {x}, y: {y} and z {z}")

            # Arbitrary point of robot at: x=372, y=317, old points (x=172, y=233)
            x_r, y_r, z_r = self.midas.get_world_coord(
                u=372,
                v=317,
                depth_map=midas_true_depth
            )
            robot_position = np.array([x_r, y_r, z_r])

            # Calculate position between human and the robot
            diff, short_dist = self.midas.calculate_points_distance(human_position, robot_position)
            if isPrint: print(f"Coordinate diff is {diff} and shortest dist is {short_dist}")

            # Calculate time
            end_time = time.time() * 1000
            time_diff = end_time - start_time
            if isPrint: print("Time needed for one frame", time_diff)

            # Update arrays
            self.elapsed_time.append(time_diff)
            self.distances_array.append(short_dist)
            self.diff_array.append(diff)

            # Show the frame with person detected
            self.yolo.draw_boxes(frame, human_boxes)
            cv2.imshow("Stream", frame)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if isPrint: print("Elapsed time", self.elapsed_time)
        cv2.destroyAllWindows()

    def plot_data(self, title=None):
        """
            Method for plotting the values
            Default name of the title is:
            :param title: 'Midas model V3.0 - dpt30_small'
        """
        if title is None:
            title = 'Midas model V3.0 - dpt30_small'
        else:
            title = title

        # First, plot the elapsed time for each frame
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.elapsed_time, marker='o', linestyle='-')
        plt.title('Elapsed Time per Frame')
        plt.xlabel('Frame')
        plt.ylabel('Time (ms)')

        # Then, plot the distances array
        plt.subplot(1, 3, 2)
        plt.plot(self.distances_array, marker='x', linestyle='-')
        plt.title('Short Distance over Frames')
        plt.xlabel('Frame')
        plt.ylabel('Distance')

        # Finally, plot the diff array
        plt.subplot(1, 3, 3)
        plt.plot(self.diff_array, marker='s', linestyle='-')
        plt.title('Differences over Frames')
        plt.xlabel('Frame')
        plt.ylabel('Difference')

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def save_data(self, name=None):
        """
        Method for saving data.
        Name is a prefix, it should contain model type and a save folder if needed.
        Example below:
        :param name: 'SavedFilesNP/dpt30_small-'
        """
        if name is None:
            elapsed_time_name = str(name) + '-elapsed_time.npy'
            distances_name = str(name) + '-distances_array.npy'
            differences_name = str(name) + '-diff_array.npy'
        else:
            elapsed_time_name = 'SavedFilesNP/dpt30_small-elapsed_time.npy'
            distances_name = 'SavedFilesNP/dpt30_small-distances_array.npy'
            differences_name = 'SavedFilesNP/dpt30_small-diff_array.npy'

        np.save(elapsed_time_name, self.elapsed_time)
        np.save(distances_name, self.distances_array)
        np.save(differences_name, self.diff_array)


def main():
    # Yolo initialized in constructor
    estimator = SafetyEstimator()
    model_path = 'estimator_include/midas_v31_models/weights/dpt_levit_224.pt'
    model_type = 'dpt_levit_224'
    estimator.init_midas_v31(model_path=model_path, v31_type=model_type)  # Model choosing, init_midas_v30() for V30
    estimator.video_stream()
    estimator.plot_data()
    # estimator.save_data()


if __name__ == '__main__':
    main()
