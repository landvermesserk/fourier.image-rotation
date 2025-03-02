import numpy as np
import cv2


def create_chessboard_image(image_size: int,
                            square_size: int=1) -> np.ndarray:
    """Creates a grayscale image that has a black and white chessboard pattern."""
    if image_size % square_size != 0:
        raise ValueError("The image size must be a multiple of the square size.")
    image = np.zeros((image_size, image_size))
    number_squares = image_size // square_size
    for row in range(number_squares):
        for col in range(number_squares):
            black_white_factor = 0 if (row + col) % 2 == 0 else 1
            image[row*square_size:(row+1)*square_size, 
                  col*square_size:(col+1)*square_size] = 255 * black_white_factor
    return image

def create_stripe_image(image_size: int,
                       square_size: int=1) -> np.ndarray:
    """Creates a grayscale image that has a black and white stripe pattern."""
    if image_size % square_size != 0:
        raise ValueError("The image size must be a multiple of the square size.")
    image = np.zeros((image_size, image_size))
    number_squares = image_size // square_size
    for row in range(number_squares):
        for col in range(number_squares):
            black_white_factor = 0 if row % 2 == 0 else 1
            image[row*square_size:(row+1)*square_size, 
                  col*square_size:(col+1)*square_size] = 255 * black_white_factor
    return image
    

def save_chessboard_image(chessboard: np.ndarray,
                          path: str="/app/samples/chessboard.jpg") -> None:
    """Save chessboard image as jpeg."""
    cv2.imwrite(path, chessboard)


# stripe_image = create_stripe_image(image_size=31, square_size=1)
# cv2.imwrite("/app/samples/stripe.jpg", stripe_image)