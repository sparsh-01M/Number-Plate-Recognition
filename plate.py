import ssl
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import easyocr

# Temporary workaround for SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context

# Load the image
image = cv2.imread("image_processing/images/test5.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
plt.show()

# Apply bilateral filter
# bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

blurred = cv2.GaussianBlur(gray, ksize=(3, 5), sigmaX=0.5) 

# Apply Canny edge detection
edges = cv2.Canny(blurred, 30, 200)

# Display edges
plt.imshow(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
plt.show()

# Find contours
key = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(key)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Initialize location variable
location = None

# Loop through contours to find a rectangle (number plate)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 8:
        location = approx
        break

# If no rectangle is found, print an error message
if location is None:
    print("Number plate contour not found")
else:
    # Draw the contours on a mask
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)

    # Display the masked image
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()

    # Crop the image to the region of the number plate
    x, y = np.where(mask == 255)
    x1, y1 = (np.min(x), np.min(y))
    x2, y2 = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    # Display the cropped image
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB))
    plt.show()

    # Use EasyOCR to read text from the cropped image
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    # Print the result
    print(result[1][1])
