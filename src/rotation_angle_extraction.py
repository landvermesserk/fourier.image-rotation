import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

index = 0
espsilon = 1e-5

image_path = f"/app/samples/bagtag_{index}.png"
image_path_fourier = f"/app/samples/bagtag_{index}_fourier_fourier_thresh.jpg"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Load the image in grayscale
image_fourier = cv2.imread(image_path_fourier, cv2.IMREAD_GRAYSCALE)

# Apply thresholding to extract bright pixels
_, binary = cv2.threshold(image_fourier, 200, 255, cv2.THRESH_BINARY)  

height, width = image_fourier.shape

y_center, x_center = height / 2, width / 2

# Find coordinates of bright pixels
y_coords, x_coords = np.where(binary == 255)

y_coords = y_coords - y_center
x_coords = x_coords - x_center

angles = np.array([np.arctan(y_coords[index] / (x_coords[index] + espsilon)) * 360 / (2 * np.pi) for index in range(len(y_coords))])
angles = angles[~np.isnan(angles)]
angles[angles < 0] += 90


# Compute the histogram
hist, bin_edges = np.histogram(angles, bins=10)  # 10 bins

# Compute bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Find the indices of the two highest counts
sorted_indices = np.argsort(hist)[::-1]  # Sort indices in descending order
max_index, second_max_index = sorted_indices[:2]  # Get top two indices

# Bin centers corresponding to the max and second max
rotation_angle_first = bin_centers[max_index]
rotation_angle_second = bin_centers[second_max_index]


if hist[second_max_index] / hist[max_index] >= 0.9:
    rotation_angle = (rotation_angle_first + rotation_angle_second) / 2

else:
    rotation_angle = rotation_angle_first

print(rotation_angle)

rotated_image = imutils.rotate_bound(image, -rotation_angle)

plt.figure(figsize=(18, 6))
plt.subplot(231), plt.hist(angles, edgecolor='black'), plt.title('Angle distribution')
plt.subplot(232), plt.imshow(image, cmap='gray'), plt.title('Original image')
plt.subplot(233), plt.imshow(rotated_image, cmap='gray'), plt.title('Rotated image')
plt.subplot(234), plt.imshow(imutils.rotate_bound(rotated_image, -90), cmap='gray'), plt.title('Rotated image by 90 degrees')
plt.subplot(235), plt.imshow(imutils.rotate_bound(rotated_image, -180), cmap='gray'), plt.title('Rotated image by 180 degrees')
plt.subplot(236), plt.imshow(imutils.rotate_bound(rotated_image, -270), cmap='gray'), plt.title('Rotated image by 270 degrees')
plt.show()


# # --------------------------Linear Regression approach-----------------------------------

# # Perform linear regression (Fit y = mx + b)
# #if len(x_coords) > 0:
# m, b = np.polyfit(x_coords, y_coords, 1)  # Linear regression

# # Define the start and end points of the line
# x_start, x_end = min(x_coords), max(x_coords)
# y_start, y_end = int(m * x_start + b), int(m * x_end + b)

# # Draw the detected line on the original image
# result_image = image_fourier.copy()
# cv2.line(result_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

# # Display the result
# plt.figure(figsize=(10, 6))
# plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
# plt.title("Dominant Line Detection")
# plt.show()
# #else:
# #    print("No bright pixels found.")


# #-----------------------------Canny and Hough Line approach------------------------------------


# # Detect edges in the spectrum
# edges = cv2.Canny(image_fourier, 50, 150)


# # Apply Hough Line Transform to detect dominant orientations
# lines = cv2.HoughLines(image_fourier, 1, np.pi / 180, threshold=100)



# # Find the dominant angle
# angles = []
# if lines is not None:
#     for rho, theta in lines[:, 0]:
#         angle = np.rad2deg(theta) - 90  # Convert radians to degrees
#         angles.append(angle)


# # Compute the most frequent angle (median for robustness)
# dominant_angle = np.median(angles) if angles else 0
# print(f"Dominant Angle: {dominant_angle:.2f} degrees")

# # Rotate the image based on the dominant angle
# def rotate_image(image, angle):
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
#     return rotated

# rotated_image = imutils.rotate_bound(image, -dominant_angle)

# # Show results
# plt.figure(figsize=(15, 5))
# plt.subplot(151), plt.imshow(image, cmap='gray'), plt.title('Original Image')
# plt.subplot(152), plt.imshow(image_fourier, cmap='gray'), plt.title('Fourier transform')
# plt.subplot(153), plt.imshow(rotated_image, cmap='gray'), plt.title('Rotated Image')
# plt.subplot(154), plt.imshow(imutils.rotate_bound(rotated_image, -90), cmap='gray'), plt.title('Rotated Image by 90 degrees')
# plt.subplot(155), plt.hist(angles, edgecolor='black'), plt.title('Angle distribution')


# plt.show()
