# https://github.com/isl-org/MiDaS/issues/268
# https://github.com/isl-org/MiDaS/issues/37
# https://github.com/isl-org/MiDaS/issues/171
# https://medium.com/@parkie0517/2d-to-3d-conversion-learning-how-to-convert-rgb-images-to-point-cloud-025a1fd77abe


import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from skimage.transform import resize  # Ensure resizing for mismatched images
import json


class midasDepth():
    def __init__(self):
        self.src_image = None
        self.depth_aligned = None
        self.measure_points = []
        self.intrinsic = None
        self.h, self.w = None, None
        self.fx, self.fy = None, None
        self.cx, self.cy = None, None

    def load_intrinsic(self, intrinsic):
        self.intrinsic = intrinsic
        self.h, self.w = self.intrinsic[0], self.intrinsic[1]
        self.fx, self.fy = self.intrinsic[2], self.intrinsic[3]  # Focal lengths
        self.cx, self.cy = self.intrinsic[4], self.intrinsic[5]  # Principal point

    def init_model(self, model="DPT_Large"):
        if model == "DPT_Large":
            self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
            self.midas.eval()
            self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        elif model == "DPT_Hybrid":
            self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            self.midas.eval()
            self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        else:
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.midas.eval()
            self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform



    def init_from_path(self, path):
        device = torch.device("cpu")
        model_type = 'dpt_swin2_large_384'
        model_path = path
        # model_loader.load_model(device, model_path=model_path,
        #                                   model_type=model_type, optimize=False)

    def inspect_model(self):
        print("State Dictionary Keys:")
        for key in self.midas.state_dict().keys():
            print(key)


    def depth_prediction(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cpu().numpy()
        depth_map = output
        return depth_map


    def get_depth_custom(self, knownPos1, knownPos2, midas_prediction):
        # normalize midas prediction to 0...1
        midas_depth_array = midas_prediction / np.max(midas_prediction)
        pixel_x_1, pixel_y_1, real_depth_1 = knownPos1
        pixel_x_2, pixel_y_2, real_depth_2 = knownPos2

        inv_d1 = 1 / real_depth_1
        inv_d2 = 1 / real_depth_2

        m1 = midas_depth_array[int(pixel_y_1), int(pixel_x_1)]
        m2 = midas_depth_array[int(pixel_y_2), int(pixel_x_2)]

        A = (inv_d2 - inv_d1) / (m2 - m1)
        B = inv_d1 - m1 * A
        print("A and B from cust function are", A, B)

        midas_prediction_aligned = 1 / (A * midas_depth_array + B)

        return midas_prediction_aligned

    def depth_to_real_lsq(self, midas_prediction, known_points, isParam=False):
        '''
            Transfer relative MiDaS depths to real depths with known points
            Args:
            midas_prediction: output from MiDaS
            known_points: points on image with known distances (x, y, distanse)
        '''

        # normalize midas prediction to 0...1
        midas_depth_array = midas_prediction / np.max(midas_prediction)

        if len(known_points) >= 2:
            # get pairs of normalized relative and real depths
            points = np.array([(midas_depth_array[int(y), int(x)], distance) for x, y, distance in known_points])

            # solve the system of equations :
            # relative_depth*(1/min_depth) + (1-relative_depth)*(1/max_depth) = 1/real_depth
            x = points[:, 0]  # normalized relative estimator_include
            y = 1 / points[:, 1]  # reversed real estimator_include
            A = np.vstack([x, 1 - x]).T

            s, t = np.linalg.lstsq(A, y, rcond=None)[0]

            min_depth = 1 / s
            max_depth = 1 / t

        else:
            print('Not enough known points to make real estimator_include estimation')
            return None

        # align relative estimator_include to real estimator_include
        A = (1 / min_depth) - (1 / max_depth)
        B = 1 / max_depth
        midas_depth_aligned = 1 / (A * midas_depth_array + B)
        print("A and B from LSQ are", A, B)

        if isParam:
            return A, B, midas_depth_array
        else:
            return midas_depth_aligned

    def convert_for_cv(self, depth_map):
        # Normalize the estimator_include map for display
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_map_normalized = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype(np.uint8)

        return depth_map_normalized

    def read_json(self, path):
        # Read data from a JSON file
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        return data

    def create_pc_rgbd(self, rgb_image, depth_map):
        """
        Creates a point cloud from an RGB-D image using Open3D.
        """

        # Convert NumPy arrays to Open3D images
        color_o3d = o3d.geometry.Image(rgb_image.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))

        # Create Open3D RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=500.0, convert_rgb_to_intensity=False)

        # Define camera intrinsics
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=self.w, height=self.h,
                                                      fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)

        # Create point cloud
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

        # Scaling x and y
        # points = np.asarray(pc.points)  # Convert Open3D point cloud to NumPy array
        # scaling_x = 2.0
        # scaling_y = 2.0  # Adjust if needed
        # points[:, 0] *= scaling_x  # Scale X
        # points[:, 1] *= scaling_y  # Scale Y
        # pc.points = o3d.utility.Vector3dVector(points)  # Convert back to Open3D format
        # Scaling x and y


        # **Ensure RGB and estimator_include images match in size**
        color_array = np.asarray(rgb_image, dtype=np.float32) / 255.0  # Normalize to [0,1]

        if color_array.shape[:2] != depth_map.shape:
            print("Warning: RGB and estimator_include image sizes do not match! Resizing RGB image.")
            color_array = resize(color_array, (depth_map.shape[0], depth_map.shape[1]), anti_aliasing=True)

        # **Assign correct colors**
        pc.colors = o3d.utility.Vector3dVector(color_array.reshape(-1, 3))

        angle_rad = np.pi  # 180 degrees in radians
        cos_theta = np.cos(angle_rad)  # -1
        sin_theta = np.sin(angle_rad)  # 0
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])

        pc.rotate(rotation_matrix, center=(0, 0, 0))

        return pc

    def visualize_point_cloud(self, pc):
        """
            Visualizes the given Open3D point cloud with color.
        """
        if isinstance(pc, o3d.geometry.PointCloud):
            if len(pc.points) == 0:
                print("Warning: The point cloud is empty!")
                return
            pcd = o3d.geometry.PointCloud()
            pcd.points = pc.points

            # Transfer colors if available
            if len(pc.colors) > 0:
                pcd.colors = pc.colors
            else:
                print("Warning: No color information available in the point cloud!")
                # Optionally add a default color if none is present
                pcd.paint_uniform_color([1, 0, 0])  # Paints all points red

            # Visualize the point cloud with Open3D
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

            o3d.visualization.draw_geometries([pcd, coord_frame])
        else:
            print("Error: The provided object is not an Open3D PointCloud.")

    def visualize_point_cloud_with_callback(self, pc):
        """Visualizes the given Open3D point cloud with an interactive callback."""
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pc)
        vis.add_geometry(coord_frame)
        vis.run()  # Blocks until window is closed
        vis.destroy_window()
        points_array = []

        self.selected_indices = vis.get_picked_points()
        if self.selected_indices:
            print("Selected Points:")
            for idx in self.selected_indices:
                point = np.asarray(pc.points)[idx]
                print(f"Index: {idx}, Coordinates: {point}")
                points_array.append(point)
                self.measure_points.append(point)

                if len(points_array) >= 2:
                    # hor_dist = np.sqrt(abs(points_array[0] ** 2 - points_array[1] ** 2) )
                    # hor_dist = np.sqrt((points_array[0] - points_array[0]) ** 2 + (points_array[1] - points_array[1]) ** 2)
                    delta_coord = (points_array[0] - points_array[1])
                    hor_dist = (delta_coord[0] ** 2 + delta_coord[1] ** 2 + delta_coord[2] ** 2) ** 0.5
                    print("Delta distance is:", delta_coord)
                    print("Horizontal distance is:", hor_dist)

        # return points_array

    def get_selected_points(self):
        if len(self.measure_points) %2 != 0:
            print("Number of selected points not even!")
            return 1

        results = []
        for i in range(0, len(self.measure_points), 2):
            diff = self.measure_points[i + 1] - self.measure_points[i]
            results.append(diff)
            short_dist = (diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2) ** 0.5
            print(f"For pair number {i} - {i+1}, delta is {diff} and shortest distance is {short_dist}")

    def get_world_coord(self, u, v, depth_map, depth_scale=1.0, isPrint=False):
        # Read the estimator_include value at (u, v)
        d = depth_map[int(v), int(u)]

        z = d / depth_scale
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        if isPrint: print(f"Points are x: {x}, y: {y} and z {z}")
        return x, y, z

    def calculate_points_distance(self, point1, point2):
        diff = point2 - point1
        short_dist = (diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2) ** 0.5
        return diff, short_dist


def main():
    midas = midasDepth()
    midas.init_model()
    # midas.init_from_path("dpt_swin2_large_384.pt")
    # midas.inspect_model()

    image = cv2.imread("calImages/image4.png")
    image_rgb = np.copy(image)
    knownPos33 = (310.5, 198.0, 150)  # 1/150 = 0.0067
    knownPos38 = (502.5, 261.75, 50)  # 1/50 = 0.02
    # knownPos_test = (502.5, 261.75, 50)  # 1/50 = 0.02
    knownPositions = [knownPos38, knownPos33]
    print("knownPositions", knownPositions)

    midas_output = midas.depth_prediction(image)
    lsq_midas_output = midas.depth_to_real_lsq(midas_prediction=midas_output, known_points=knownPositions)

    cv2.imshow("Original image", image)
    cv2.imshow("Midas raw output", midas.convert_for_cv(midas_output))
    cv2.imshow("Calibrated with LSQ midas output", lsq_midas_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_json():
    midas = midasDepth()
    midas.init_model()
    image = cv2.imread("SruvCameraData/3. Frame_RGB_2025-02-22_17-49-24.png")
    depth_image = cv2.imread("SruvCameraData/3. Frame_MiDaS_DPT_Large_2025-02-22_17-49-24.png")

    xy_path = 'SruvCameraData/coordinates.json'
    depth_path = 'SruvCameraData/distances.json'
    coordinates = midas.read_json(xy_path)
    depth_coordinates = midas.read_json(depth_path)
    calibration_points = [(coord['x'], coord['y'], depth['d'])
                for coord, depth in zip(coordinates, depth_coordinates)]
    print("Calibration points are:")
    print(calibration_points)

    midas_output = midas.depth_prediction(image)
    lsq_midas_output = midas.depth_to_real_lsq(midas_prediction=midas_output, known_points=calibration_points)

    cv2.imshow("Original image", image)
    cv2.imshow("Midas raw output",  midas.convert_for_cv(midas_output))
    cv2.imshow("Calibrated with LSQ midas output", midas.convert_for_cv(lsq_midas_output))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Surveillance camera
    h, w = midas_output.shape
    # fx, fy = 2.17404891e+03, 2.20471536e+03
    fx, fy = 2.17404891e+03 * 0.5, 2.20471536e+03 * 0.5
    cx, cy = 3.17695080e+02, 2.37041115e+02

    # fx, fy = 2.91010521e+03, 2.73899617e+03
    # cx, cy = 6.52220285e+02, 3.66729543e+02

    intrinsic_param_cust = (h, w, fx, fy, cx, cy)
    midas.load_intrinsic(intrinsic_param_cust)
    print("intrinsic_param_cust[0]", intrinsic_param_cust[0])
    pc = midas.create_pc_rgbd(rgb_image=image, depth_map=lsq_midas_output)
    midas.visualize_point_cloud_with_callback(pc)
    # midas.visualize_point_cloud(pc)

    midas.get_world_coord(u=372, v=317, depth_map=lsq_midas_output, isPrint=True)
    midas.get_selected_points()





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # main()
    test_json()