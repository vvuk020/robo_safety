# RoboSafety Depth Estimator

A Python-based safety estimation system using **YOLOv8** for human detection and **MiDaS v3.0 / v3.1** for depth estimation. Designed for robotics and surveillance to calculate distances between humans and robots in video streams.

---

## Features

* Human detection using YOLOv8
* Depth estimation with MiDaS v3.0 and v3.1 models
* Calibration for accurate real-world distance calculation
* Video stream processing with live detection and depth measurement
* Plotting of processing time and distance metrics

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/vvuk020/RoboSafety_DepthEstm_clean.git
cd RoboSafety_DepthEstm_clean
```

---

## Usage

Run the main safety estimator:

```bash
python cam_safety_alg_video_optm.py
```

By default, it will:

* Load **MiDaS v3.1 model**
* Calibrate with pre-set points
* Process `SavedFilesNP/Video.avi` stream
* Display live detection with YOLO boxes
* Show plots of processing time and distances

---

### Changing Depth Model

To use MiDaS v3.0:

```python
estimator.init_midas_v30(model_name="DPT_Large")
```

To use a custom MiDaS v3.1 model:

```python
estimator.init_midas_v31(
    model_path='estimator_include/midas_v31_models/weights/dpt_levit_224.pt',
    v31_type='dpt_levit_224'
)
```

---

## Folder Structure

```
estimator_include/      # Core modules and models
SavedFilesNP/           # Saved video and result arrays
CameraCalibration-main/ # Calibration tools
cam_safety_alg_video_optm.py # Main entry script
```

---

## License

MIT License — free to use and modify.

---