import cv2
import numpy as np

# OpenCV provides the calcHist functions to calculate the histogram of images
def calculate_histogram(image):
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate the histogram for the hue channel
    histogram = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    
    return histogram

# OpenCV provides functions such as filter2D to apply a kernel to an image
def calculate_gradient_orientation(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sobel operators to find the x and y gradients
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and orientation
    magnitude = cv2.magnitude(grad_x, grad_y)
    orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    
    # Normalize for display purposes
    gradient_magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    gradient_orientation = cv2.normalize(orientation, None, 0, 1, cv2.NORM_MINMAX)
    
    return gradient_magnitude, gradient_orientation

def masked_orientations(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate gradients using Sobel operator
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude and orientation
    magnitude = cv2.magnitude(grad_x, grad_y)
    orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    # Normalize for display
    normalized_magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    normalized_orientation = cv2.normalize(orientation, None, 0, 1, cv2.NORM_MINMAX)

    # Threshold to create a binary mask of the orientation
    # Here we are assuming that significant gradients are those that are above average magnitude.
    threshold = np.mean(magnitude)
    _, mask = cv2.threshold(magnitude, threshold, 1, cv2.THRESH_BINARY)

    # Convert mask to an 8-bit image
    mask = mask.astype(np.uint8)

    # Use the mask to create masked orientation image
    # Select between normalized orientation and a red background
    masked_orientation = cv2.bitwise_and(normalized_orientation, normalized_orientation, mask=mask)

    return masked_orientation



