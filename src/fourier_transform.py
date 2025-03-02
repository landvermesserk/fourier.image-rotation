import numpy as np
import cv2
import argparse
import os
import time
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
formatter = logging.Formatter("{asctime} - {levelname} - {message}", 
                              style="{",datefmt="%Y-%m-%d %H:%M",)
console_handler.setFormatter(formatter)


def time_it(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure execution time of a function."""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time: float = time.time()
        result: Any = func(*args, **kwargs)
        end_time: float = time.time()
        logger.info(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def create_fourier_transform(image: np.ndarray) -> np.ndarray:
    """Computes the discrete fourier transform of a two dimensional image."""
    chessboard_fourier = np.fft.fft2(image)
    fshift = np.fft.fftshift(chessboard_fourier)
    return 20 * np.log(np.abs(fshift))

def apply_threshold_to_image(image: np.ndarray, 
                             pixel_threshold: int=127) -> np.ndarray:
    """Based on the threshold set low intensity pixels to zero and all others to 255."""
    _, thresh1 = cv2.threshold(image, pixel_threshold, 255, cv2.THRESH_BINARY)
    return thresh1


@time_it
def apply_and_save_fourier_transform(image_path: str) -> str:
    """Creates fourier transform and saves the image."""
    if not os.path.exists(image_path):
        raise ValueError("The specified image path does not exist.")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    magnitude_spectrum = create_fourier_transform(image=image)
    file_name = os.path.basename(image_path)
    file_name = "_".join([file_name.split(".")[0], "fourier"])
    image_path_fourier = f"/app/samples/{file_name}.jpg"

    cv2.imwrite(image_path_fourier, magnitude_spectrum)

    return image_path_fourier


def apply_and_save_fourier_transform_threshold(image_path_fourier: str,
                                               pixel_threshold: int=127) -> str:
    """Creates fourier transform and saves the image."""
    if not os.path.exists(image_path_fourier):
        raise ValueError("The specified image path does not exist.")
    image = cv2.imread(image_path_fourier, cv2.IMREAD_GRAYSCALE)
    image = apply_threshold_to_image(image=image, 
                                     pixel_threshold=pixel_threshold)
    file_name = os.path.basename(image_path_fourier)
    file_name = "_".join([file_name.split(".")[0], "fourier"])
    image_path_fourier = f"/app/samples/{file_name}_thresh.jpg"

    cv2.imwrite(image_path_fourier, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', help='The image path.')
    parser.add_argument('--pixel_threshold', help='The fourier postprocessing threshold.')
    args = parser.parse_args()
    image_path = args.image_path
    pixel_threshold = int(args.pixel_threshold)

    image_path_fourier = apply_and_save_fourier_transform(image_path=image_path)
    apply_and_save_fourier_transform_threshold(image_path_fourier=image_path_fourier, 
                                               pixel_threshold=pixel_threshold)




