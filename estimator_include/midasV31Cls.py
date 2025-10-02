import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
midas_models_path = os.path.join(current_dir, 'midas_v31_models')
sys.path.append(midas_models_path)

import torch
import cv2
import numpy as np
from midas_v31_models.model_loader import load_model
from midasV30Cls import midasDepth



class midasV31(midasDepth):
    def __init__(self, device=None, model_path=None, model_type=None, optimize=False, height=None, square=False, use_camera=False):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model_type = model_type
        self.optimize = optimize
        self.height = height
        self.square = square
        self.use_camera = use_camera
        self.model = None
        self.transform = None
        self.net_w = None
        self.net_h = None

    def load_model(self):
        print(f"Using device: {self.device}")
        self.model, self.transform, self.net_w, self.net_h = load_model(
            self.device,
            self.model_path,
            self.model_type,
            self.optimize,
            self.height,
            self.square
        )
        self.model.eval()  # Set model to evaluation mode

    def process(self, image):
        """
        Run the inference and interpolate.

        Args:
            image: the input image for the neural network
            input_size: the size (width, height) of the neural network input
            target_size: the size (width, height) the neural network output is interpolated to

        Returns:
            The prediction
        """
        input_size = (self.net_w, self.net_h)
        original_image_rgb = image
        image = self.transform({"image": original_image_rgb})["image"]
        target_size = original_image_rgb.shape[1::-1]

        if "openvino" in self.model_type:
            if not self.use_camera:
                print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")

            sample = [np.reshape(image, (1, 3, *input_size))]
            prediction = self.model(sample)[self.model.output(0)][0]
            prediction = cv2.resize(prediction, dsize=target_size, interpolation=cv2.INTER_CUBIC)
        else:
            sample = torch.from_numpy(image).to(self.device).unsqueeze(0)

            if self.optimize and self.device.type == "cuda":
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite estimator_include values to some extent for\n"
                      "  half-floats.")
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            if not self.use_camera:
                height, width = sample.shape[2:]
                print(f"    Input resized to {width}x{height} before entering the encoder")

            with torch.no_grad():
                prediction = self.model(sample)
                prediction = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=target_size[::-1],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )

        return prediction

    def try_process(self, image_path):
        image = cv2.imread(image_path)
        original_image_rgb = image
        image = self.transform({"image": original_image_rgb})["image"]

        print("Performing inference...")
        prediction = self.process(
            image,
            (self.net_w, self.net_h),
            original_image_rgb.shape[1::-1]
        )

        depth_min = prediction.min()
        depth_max = prediction.max()
        depth_map_normalized = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype(np.uint8)

        print("Prediction: ", prediction)
        cv2.imshow("original img", original_image_rgb)
        cv2.imshow("estimator_include", depth_map_normalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def try_midas():
    # model_path = 'midas_v31_models/weights/dpt_swin2_large_384.pt'
    # model_type = 'dpt_swin2_large_384'
    # model_path = 'midas_v31_models/weights/dpt_levit_224.pt'
    # model_type = 'dpt_levit_224'

    model = midasV31(
        model_path='midas_v31_models/weights/dpt_swin2_tiny_256.pt',
        model_type='dpt_swin2_tiny_256'
    )
    print("device is ", model.device)
    model.load_model()
    img = cv2.imread('midas_v31_models/image4.png')
    prediction = model.process(img)

    depth_min = prediction.min()
    depth_max = prediction.max()
    depth_map_normalized = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype(np.uint8)
    print("Prediction: ", prediction)
    cv2.imshow("original img", img)
    cv2.imshow("estimator_include", depth_map_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # model_path = 'midas_v31_models/weights/dpt_swin2_large_384.pt'
    # model_type = 'dpt_swin2_large_384'
    # model_path = 'midas_v31_models/weights/dpt_levit_224.pt'
    # model_type = 'dpt_levit_224'

    # processor = midasV31(
    #     model_path='midas_v31_models/weights/dpt_swin2_tiny_256.pt',
    #     model_type='dpt_swin2_tiny_256'
    # )
    # processor.load_model()
    # processor.try_process('midas_v31_models/image4.png')
    try_midas()
