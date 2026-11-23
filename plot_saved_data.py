
import numpy as np
from cam_safety_alg_video_optm import SafetyEstimator


def main():

    # Initialize estimator (only to use its plotting method)
    estimator = SafetyEstimator()

    # Select model
    model_type = 'dpt_levit_224'

    # Load previously saved NumPy data
    estimator.elapsed_time = np.load(
        f'SavedFilesNP/{model_type}-elapsed_time.npy'
    ).tolist()

    estimator.distances_array = np.load(
        f'SavedFilesNP/{model_type}-distances_array.npy'
    ).tolist()

    estimator.diff_array = np.load(
        f'SavedFilesNP/{model_type}-diff_array.npy'
    ).tolist()

    # Plot using your original class method
    estimator.plot_data(title=f'MiDaS model - {model_type}')


if __name__ == '__main__':
    main()
